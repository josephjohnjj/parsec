#include "parsec/mca/device/cuda/device_cuda_migrate.h"

extern int parsec_device_cuda_enabled;
extern int parsec_cuda_migrate_chunk_size; // chunks of task migrated to a device (default=5)
extern int parsec_migrate_statistics; 

parsec_device_cuda_info_t *device_info;
static parsec_list_t *migrated_task_list;           // list of all migrated task
static int NDEVICES;                                // total number of GPUs
static parsec_hash_table_t *task_mapping_ht = NULL; // hashtable for storing task mapping
static int task_migrated_per_tp = 0;
static int tp_count;

double start = 0;
double end = 0;

PARSEC_OBJ_CLASS_INSTANCE(migrated_task_t, parsec_list_item_t, NULL, NULL);

static parsec_key_fn_t task_mapping_table_generic_key_fn = {
    .key_equal = parsec_hash_table_generic_64bits_key_equal,
    .key_hash = parsec_hash_table_generic_64bits_key_hash,
    .key_print = parsec_hash_table_generic_64bits_key_print};

int parsec_gpu_task_count_start;
int parsec_gpu_task_count_end;

static void task_mapping_ht_free_elt(void *_item, void *table)
{
    task_mapping_item_t *item = (task_mapping_item_t *)_item;
    parsec_key_t key = item->ht_item.key;
    parsec_hash_table_nolock_remove(table, key);
    free(item);
}

static void gpu_dev_profiling_init()
{
    parsec_profiling_add_dictionary_keyword("GPU_TASK_COUNT", "fill:#FF0000",
        sizeof(gpu_dev_prof_t),
        "first_queue_time{double};select_time{double};second_queue_time{double};exec_time{double};stage_in_time{double};device_index{int32_t};task_count{int32_t};waiting_tasks{int32_t};type{int32_t}",
        &parsec_gpu_task_count_start, &parsec_gpu_task_count_end);
}

/**
 * @brief The function initialises the data structures required
 * for inter-device migration.
 *
 * @param ndevices number of devices
 * @return int
 */

int parsec_cuda_migrate_init(int ndevices)
{
    int i, j;
    start = MPI_Wtime();

    NDEVICES = ndevices;
    device_info = (parsec_device_cuda_info_t *)calloc(ndevices, sizeof(parsec_device_cuda_info_t));
    migrated_task_list = PARSEC_OBJ_NEW(parsec_list_t);

    for (i = 0; i < NDEVICES; i++)
    {
        for (j = 0; j < EXECUTION_LEVEL; j++) device_info[i].task_count[j] = 0;
        device_info[i].load = 0;
        device_info[i].level0 = 0;
        device_info[i].level1 = 0;
        device_info[i].level2 = 0;
        device_info[i].total_tasks_executed = 0;
        device_info[i].received = 0;
        device_info[i].last_device = i;
    }

    task_mapping_ht = PARSEC_OBJ_NEW(parsec_hash_table_t);
    parsec_hash_table_init(task_mapping_ht, offsetof(task_mapping_item_t, ht_item), 16, task_mapping_table_generic_key_fn, NULL);

#if defined(PARSEC_PROF_TRACE)
    gpu_dev_profiling_init();
#endif

    return 0;
}

int parsec_cuda_migrate_fini()
{
    int i;

    end = MPI_Wtime();

#if defined(PARSEC_HAVE_CUDA)
    nvmlShutdown();
#endif

    parsec_hash_table_for_all(task_mapping_ht, task_mapping_ht_free_elt, task_mapping_ht);
    parsec_hash_table_fini(task_mapping_ht);
    PARSEC_OBJ_RELEASE(task_mapping_ht);
    task_mapping_ht = NULL;

    if(parsec_migrate_statistics)
    {
        for (i = 0; i < NDEVICES; i++)
        {
            printf("\n*********** DEVICE %d *********** \n", i);
            printf("Total tasks executed: %d \n", device_info[i].total_tasks_executed);
            printf("Tasks migrated      : level0 %d, level1 %d, level2 %d (Total %d)\n",
                   device_info[i].level0, device_info[i].level1, device_info[i].level2,
                   device_info[i].level0 + device_info[i].level1 + device_info[i].level2);
            printf("Task check          : level0 %d level1 %d level2 %d total %d \n",
                   parsec_cuda_get_device_task(i, 0),
                   parsec_cuda_get_device_task(i, 1),
                   parsec_cuda_get_device_task(i, 2),
                   parsec_cuda_get_device_task(i, -1));
            printf("Task received       : %d \n", device_info[i].received);
        }
    }
    printf("\n---------Execution time = %lf ------------ \n", end - start);
    PARSEC_OBJ_RELEASE(migrated_task_list);
    free(device_info);

    return 0;
}

double current_time()
{
    return (MPI_Wtime() - start);
}

/**
 * @brief returns the number of tasks in a particular device
 *
 * @param device index of the device
 * @param level level of execution
 * @return int
 */

int parsec_cuda_get_device_task(int device, int level)
{
    if (level == -1)
        return (device_info[device].task_count[0] +
                device_info[device].task_count[1] +
                device_info[device].task_count[2]);

    return device_info[device].task_count[level];
}

/**
 * @brief sets the number of tasks in a particular device
 *
 * @param device index of the device
 * @param level level of execution
 * @return int
 */

int parsec_cuda_set_device_task(int device, int task_count, int level)
{
    int rc = parsec_atomic_fetch_add_int32(&(device_info[device].task_count[level]), task_count);
    return rc + task_count;
}

/**
 * @brief Incerement the total task executed by a device
 *
 * @param device index of the device
 * @return int
 */

int parsec_cuda_tasks_executed(int device)
{
    int rc = parsec_atomic_fetch_add_int32(&(device_info[device].total_tasks_executed), 1);
    return rc + 1;
}

/**
 * @brief returns 1 if the device is starving, 0 if its is not
 *
 * @param device index number of the device
 * @return int
 *
 */
int is_starving(int device)
{
    /**
     * @brief The default number of execution stream in PaRSEC is 2. We assume
     * starvtion if the number of ready tasks available is less than twice the
     * number of execution stream.
     */
    parsec_device_gpu_module_t* d = parsec_mca_device_get( DEVICE_NUM(device) );
    //return (parsec_cuda_get_device_task(device, -1) < 5) ? 1 : 0;
    return (d->mutex < 5) ? 1 : 0;
}

int will_starve(int device)
{
    /**
     * @brief The default number of execution stream in PaRSEC is 2. We assume
     * starvtion if migrating a task will push the number of ready tasks available
     * to less than twice the number of execution stream.
     */
    parsec_device_gpu_module_t* d = parsec_mca_device_get( DEVICE_NUM(device) );
    //return ((parsec_cuda_get_device_task(device, -1) - 1) < 5) ? 1 : 0;
    return (d->mutex < 5) ? 1 : 0;

}

/**
 * @brief returns the index of a starving device and returns -1
 * if no device is starving.
 *
 * @param dealer_device device probing for a starving device
 * @param ndevice total number of devices
 * @return int
 *
 */
int find_starving_device(int dealer_device)
{
    int i = 0;
    int starving_device = 0;
    int next_device = ((device_info[dealer_device].last_device) + 1) % NDEVICES;
    int final_device = next_device + NDEVICES;

    // use a round robin method to find starving device
    for (i = next_device; i < final_device; i++)
    {
        starving_device = i % NDEVICES;

        if (starving_device == dealer_device)
            continue;

        if (is_starving(starving_device))
            return starving_device;
    }

    return -1;
}

/**
 * @brief This function will be called in __parsec_context_wait() just before
 * parsec_current_scheduler->module.select(). This will ensure that the migrated tasks
 * will get priority over new tasks.
 *
 * When a compute thread calls this function, it is forced to try to be a manager of the
 * a device. If the device already has a manager, the compute thread passes the control of
 * the task to the manager. If not the compute thread will become the manager.
 *
 * @param es
 * @return int
 */

int parsec_cuda_mig_task_dequeue(parsec_execution_stream_t *es)
{
    char tmp[128];
    migrated_task_t *mig_task = NULL;
    parsec_gpu_task_t *migrated_gpu_task = NULL;
    parsec_device_gpu_module_t *dealer_device = NULL;
    parsec_device_gpu_module_t *starving_device = NULL;
    int stage_in_status = 0;

    mig_task = (migrated_task_t *)parsec_list_try_pop_front(migrated_task_list);

    if (mig_task != NULL)
    {
        PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t *)mig_task);
        migrated_gpu_task = mig_task->gpu_task;
        assert(migrated_gpu_task->migrate_status != TASK_NOT_MIGRATED);
        dealer_device = mig_task->dealer_device;
        starving_device = mig_task->starving_device;
        stage_in_status = mig_task->stage_in_status;

        change_task_features(migrated_gpu_task, dealer_device, starving_device, stage_in_status);

        PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t *)migrated_gpu_task);
        parsec_atomic_fetch_inc_int32(&device_info[CUDA_DEVICE_NUM(starving_device->super.device_index)].received);
        parsec_cuda_kernel_scheduler(es, (parsec_gpu_task_t *)migrated_gpu_task, starving_device->super.device_index);
        PARSEC_OBJ_DESTRUCT(mig_task);
        free(mig_task);

        return 1;
    }

    return 0;
}

/**
 * This function enqueues the migrated task to a node level queue.
 *
 *  Returns: negative number if any error occured.
 *           positive: starving device index.
 */
int parsec_cuda_mig_task_enqueue(parsec_execution_stream_t *es, migrated_task_t *mig_task)
{
    parsec_list_push_back((parsec_list_t *)migrated_task_list, (parsec_list_item_t *)mig_task);

    parsec_gpu_task_t *migrated_gpu_task = mig_task->gpu_task;
    parsec_device_gpu_module_t *starving_device = mig_task->starving_device;
    char tmp[MAX_TASK_STRLEN];
    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream, "Enqueue task %s to device queue %d", parsec_task_snprintf(tmp, MAX_TASK_STRLEN, ((parsec_gpu_task_t *)migrated_gpu_task)->ec), CUDA_DEVICE_NUM(starving_device->super.device_index));

    (void)es;
    return 0;
}

/**
 * @brief check if there are any devices starving. If there are any starving device migrate
 * task from the dealer device to the starving device.
 *
 * @param es
 * @param dealer_gpu_device
 * @return int
 */

int migrate_to_starving_device(parsec_execution_stream_t *es, parsec_device_gpu_module_t *dealer_device)
{
    int starving_device_index = -1, dealer_device_index = 0;
    int nb_migrated = 0, execution_level = 0, stream_index = 0;
    int i = 0, j = 0, k = 0, d = 0;
    parsec_gpu_task_t *migrated_gpu_task = NULL;
    parsec_device_gpu_module_t *starving_device = NULL;
    migrated_task_t *mig_task = NULL;

    dealer_device_index = CUDA_DEVICE_NUM(dealer_device->super.device_index);
    if (will_starve(dealer_device_index))
        return 0;

    // parse all available device looking for starving devices.
    int d_first = (device_info[dealer_device_index].last_device + 1) % NDEVICES;
    for (d = d_first; d < (d_first + NDEVICES); d++)
    {
        starving_device_index = d % NDEVICES;
        if (d == dealer_device_index || !(is_starving(starving_device_index)))
            continue;

        starving_device = (parsec_device_gpu_module_t *)parsec_mca_device_get(DEVICE_NUM(starving_device_index));

        for (i = 0; i < parsec_cuda_migrate_chunk_size; i++)
        {
            /**
             * @brief Tasks are searched in different levels one by one. At this point we assume
             * that the cost of migration increases, as the level increase.
             */

            // level 0 - task is just pushed to the device queue
            migrated_gpu_task = (parsec_gpu_task_t *)parsec_list_try_pop_back(&(dealer_device->pending)); // level 0
            execution_level = 0;
            if (migrated_gpu_task == NULL)
            {
                // level1 - task is aavailble in the stage_in queue. Stage_in not started.
                migrated_gpu_task = (parsec_gpu_task_t *)parsec_list_try_pop_back(dealer_device->exec_stream[0]->fifo_pending); // level 1
                execution_level = 1;

                if (migrated_gpu_task == NULL)
                {
                    for (j = 0; j < (dealer_device->max_exec_streams - 2); j++)
                    {
                        // level2 - task is available in one of the execution queue stage_in is complete
                        migrated_gpu_task = (parsec_gpu_task_t *)parsec_list_try_pop_back(dealer_device->exec_stream[(2 + j)]->fifo_pending); // level2
                        if (migrated_gpu_task != NULL)
                        {
                            execution_level = 2;
                            stream_index = 2 + j;
                            break;
                        }
                    }
                }
            }

            if (migrated_gpu_task != NULL)
            {
                assert(migrated_gpu_task->ec != NULL);
                PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t *)migrated_gpu_task);
                /**
                 * @brief if the task is a not a computational kerenel or if it is a task that has
                 * already been migrated, we stop the migration and push it back to the queue.
                 */
                if (migrated_gpu_task->task_type != PARSEC_GPU_TASK_TYPE_KERNEL || migrated_gpu_task->migrate_status > TASK_NOT_MIGRATED)
                {
                    if (execution_level == 0)
                    {
                        parsec_list_push_back(&(dealer_device->pending), (parsec_list_item_t *)migrated_gpu_task);
                        migrated_gpu_task = NULL;
                    }
                    else if (execution_level == 1)
                    {
                        parsec_list_push_back(dealer_device->exec_stream[0]->fifo_pending, (parsec_list_item_t *)migrated_gpu_task);
                        migrated_gpu_task = NULL;
                    }
                    else if (execution_level == 2)
                    {
                        parsec_list_push_back(dealer_device->exec_stream[stream_index]->fifo_pending, (parsec_list_item_t *)migrated_gpu_task);
                        migrated_gpu_task = NULL;
                    }
                }

                if (migrated_gpu_task != NULL) // make sure the task was not returned to the queue
                {
                    if(parsec_migrate_statistics)
                    {
                        if (execution_level == 0)
                        {
                            parsec_cuda_set_device_task(dealer_device_index, /* count */ -1, /* level */ 0);
                            device_info[dealer_device_index].level0++;
                        }
                        if (execution_level == 1)
                        {
                            parsec_cuda_set_device_task(dealer_device_index, /* count */ -1, /* level */ 1);
                            device_info[dealer_device_index].level1++;
                        }
                        if (execution_level == 2)
                        {
                            parsec_cuda_set_device_task(dealer_device_index, /* count */ -1, /* level */ 2);
                            device_info[dealer_device_index].level2++;
                        }
                    }
                    nb_migrated++;
                    parsec_atomic_fetch_inc_int32(&task_migrated_per_tp);

                    /**
                     * @brief change migrate_status according to the status of the stage in of the
                     * stage_in data.
                     */
                    if (execution_level == 2)
                        migrated_gpu_task->migrate_status = TASK_MIGRATED_AFTER_STAGE_IN;
                    else
                        migrated_gpu_task->migrate_status = TASK_MIGRATED_BEFORE_STAGE_IN;

                    /**
                     * @brief An object of type migrated_task_t is created store the migrated task
                     * and other associated details. This object is enqueued to a node level queue.
                     * The main objective of this was to make sure that the manager does not have to sepend
                     * time on migration. It can select the task for migration, enqueue it to the node level
                     * queue and then return to its normal working.
                     */
                    mig_task = (migrated_task_t *)calloc(1, sizeof(migrated_task_t));
                    PARSEC_OBJ_CONSTRUCT(mig_task, parsec_list_item_t);

                    mig_task->gpu_task = migrated_gpu_task;
                    for (k = 0; k < MAX_PARAM_COUNT; k++) migrated_gpu_task->candidate[i] = NULL;
                    mig_task->dealer_device = dealer_device;
                    mig_task->starving_device = starving_device;
                    mig_task->stage_in_status = (execution_level == 2) ? TASK_MIGRATED_AFTER_STAGE_IN : TASK_MIGRATED_BEFORE_STAGE_IN;

                #if defined(PARSEC_PROF_TRACE)
                    migrated_gpu_task->select_time = MPI_Wtime();
                #endif
                    PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t *)mig_task);
                    parsec_cuda_mig_task_enqueue(es, mig_task);

                    device_info[dealer_device_index].last_device = starving_device_index;

                    char tmp[MAX_TASK_STRLEN];
                    PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream, "Task %s migrated (level %d, stage_in %d) from device %d to device %d: nb_migrated %d",
                                         parsec_task_snprintf(tmp, MAX_TASK_STRLEN, ((parsec_gpu_task_t *)migrated_gpu_task)->ec),
                                         execution_level, mig_task->stage_in_status, dealer_device_index, starving_device_index, nb_migrated);
                }
            }

            if (will_starve(dealer_device_index))
                break;
        } // end for i

        if (will_starve(dealer_device_index))
            break;
    } // end for d

    migrated_gpu_task = NULL;
    /* update the expected load on the GPU device */
    parsec_device_load[dealer_device->super.device_index] -= nb_migrated * parsec_device_sweight[dealer_device->super.device_index];
    return nb_migrated;
}

/**
 * @brief This function changes the features of a task, in a way that it is preped
 * for migration.
 *
 * @param gpu_task
 * @param dealer_device
 * @param stage_in_status
 * @return int
 */

int change_task_features(parsec_gpu_task_t *gpu_task, parsec_device_gpu_module_t *dealer_device,
                         parsec_device_gpu_module_t *starving_device, int stage_in_status)
{
    int i = 0;
    parsec_task_t *task = gpu_task->ec;
    parsec_data_copy_t *src_copy = NULL;
    char tmp[128];

    /**
     * Data is already staged in the dealer device and we can find all the data
     * of the tasks to be migrated in the dealer device.
     */
    if (stage_in_status == TASK_MIGRATED_AFTER_STAGE_IN)
    {

        for (i = 0; i < task->task_class->nb_flows; i++)
        {
            if (task->data[i].data_out == NULL)
                continue;
            if (PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & gpu_task->flow[i]->flow_flags)) // CTL flow
                continue;

            parsec_data_t *original = task->data[i].data_out->original;
            parsec_atomic_lock(&original->lock);

            assert(original->device_copies[dealer_device->super.device_index] != NULL);
            assert(original->device_copies[dealer_device->super.device_index] == task->data[i].data_out);
            assert(task->data[i].data_out->device_index == dealer_device->super.device_index);

            /**
             * Even if the task has only read access, the data may have been modified
             * by another task, and it may be 'dirty'. We check the version of the data
             * to verify if it is dirty. If it is, then it is pushed to gpu_mem_owned_lru,
             * if not is is pused to gpu_mem_lru.
             */
            if ((PARSEC_FLOW_ACCESS_READ & gpu_task->flow[i]->flow_flags) &&
                !(PARSEC_FLOW_ACCESS_WRITE & gpu_task->flow[i]->flow_flags))
            {
                assert(task->data[i].data_out->readers > 0);
                /**
                 * we set a possible candidate for this flow of the task. This will allow
                 * us to easily find the stage_in data as the possible candidate in
                 * parsec_gpu_data_stage_in() function.
                 */
                gpu_task->candidate[i] = task->data[i].data_out;

                parsec_list_item_ring_chop((parsec_list_item_t *)task->data[i].data_out);
                PARSEC_LIST_ITEM_SINGLETON(task->data[i].data_out);

                if (original->device_copies[0] == NULL || task->data[i].data_out->version > original->device_copies[0]->version)
                {
                    task->data[i].data_out->coherency_state = PARSEC_DATA_COHERENCY_OWNED;
                    parsec_list_push_back(&dealer_device->gpu_mem_owned_lru, (parsec_list_item_t *)task->data[i].data_out);
                }
                else
                {
                    task->data[i].data_out->coherency_state = PARSEC_DATA_COHERENCY_SHARED;
                    parsec_list_push_back(&dealer_device->gpu_mem_lru, (parsec_list_item_t *)task->data[i].data_out);
                }
            }
            /**
             * If the task has only read-write access, the data may have been modified
             * by another task, and it may be 'dirty'. We check the version of the data
             * to verify if it is dirty. If it is, then it is pushed to gpu_mem_owned_lru,
             * if not is is pused to gpu_mem_lru.
             */
            if ((PARSEC_FLOW_ACCESS_READ & gpu_task->flow[i]->flow_flags) &&
                (PARSEC_FLOW_ACCESS_WRITE & gpu_task->flow[i]->flow_flags))
            {
                assert(task->data[i].data_out->readers > 0);
                assert(original->device_copies[0] != NULL);
                assert(task->data[i].data_in == original->device_copies[0]);
                /**
                 * we set a possible candidate for this flow of the task. This will allow
                 * us to easily find the stage_in data as the possible candidate in
                 * parsec_gpu_data_stage_in() function.
                 */
                gpu_task->candidate[i] = task->data[i].data_out;

                parsec_list_item_ring_chop((parsec_list_item_t *)task->data[i].data_out);
                PARSEC_LIST_ITEM_SINGLETON(task->data[i].data_out);

                if (original->device_copies[0] == NULL || task->data[i].data_out->version > original->device_copies[0]->version)
                {
                    task->data[i].data_out->coherency_state = PARSEC_DATA_COHERENCY_OWNED;
                    parsec_list_push_back(&dealer_device->gpu_mem_owned_lru, (parsec_list_item_t *)task->data[i].data_out);
                }
                else
                {
                    task->data[i].data_out->coherency_state = PARSEC_DATA_COHERENCY_SHARED;
                    parsec_list_push_back(&dealer_device->gpu_mem_lru, (parsec_list_item_t *)task->data[i].data_out);
                }
            }
            /**
             * If the flow is write only, we free the data immediatly as this data should never
             * be written back. As the data_in of a write only flow is always CPU copy we revert
             * to the original stage_in mechanism for write only flows.
             */
            if (!(PARSEC_FLOW_ACCESS_READ & gpu_task->flow[i]->flow_flags) &&
                (PARSEC_FLOW_ACCESS_WRITE & gpu_task->flow[i]->flow_flags))
            {
                assert(task->data[i].data_out->readers == 0);
                assert(task->data[i].data_out->super.super.obj_reference_count == 1);
                assert(original->device_copies[0] != NULL);
                assert(task->data[i].data_in == original->device_copies[0]);

                parsec_list_item_ring_chop((parsec_list_item_t *)task->data[i].data_out);
                PARSEC_LIST_ITEM_SINGLETON(task->data[i].data_out);
                
                parsec_device_gpu_module_t *gpu_device = (parsec_device_gpu_module_t *)parsec_mca_device_get(task->data[i].data_out->device_index);
                parsec_data_copy_detach(original, task->data[i].data_out, gpu_device->super.device_index);
                PARSEC_OBJ_RELEASE(task->data[i].data_out);
                zone_free(gpu_device->memory, (void *)(task->data[i].data_out->device_private));
            }

            parsec_atomic_unlock(&original->lock);
            
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                 "Migrate: data %p attached to original %p [readers %d, ref_count %d] migrated from device %d to %d (stage_in: %d)",
                                 task->data[i].data_out, original, task->data[i].data_out->readers,
                                 task->data[i].data_out->super.super.obj_reference_count, dealer_device->super.device_index,
                                 starving_device->super.device_index, TASK_MIGRATED_AFTER_STAGE_IN);
        }
    }

    return 0;
}

int gpu_data_version_increment(parsec_gpu_task_t *gpu_task, parsec_device_gpu_module_t *gpu_device)
{
    int i;
    parsec_task_t *task = gpu_task->ec;

    for (i = 0; i < task->task_class->nb_flows; i++)
    {
        if (task->data[i].data_out == NULL)
            continue;
        if (PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & gpu_task->flow[i]->flow_flags)) // CTL flow
            continue;

        if ((PARSEC_FLOW_ACCESS_WRITE & gpu_task->flow[i]->flow_flags) && (gpu_task->task_type != PARSEC_GPU_TASK_TYPE_PREFETCH))
        {
            parsec_gpu_data_copy_t *gpu_elem = task->data[i].data_out;
            gpu_elem->version++; /* on to the next version */
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                                 "GPU[%s]: GPU copy %p [ref_count %d] increments version to %d at %s:%d",
                                 gpu_device->super.name,
                                 gpu_elem, gpu_elem->super.super.obj_reference_count, gpu_elem->version,
                                 __FILE__, __LINE__);
        }
    }

    return 0;
}

/**
 * @brief Associate a task with a particular device_index.
 *
 * @param task
 * @param device_index
 * @return int
 */
int update_task_to_device_mapping(parsec_task_t *task, int device_index)
{
    parsec_key_t key;
    task_mapping_item_t *item;

    key = task->task_class->make_key(task->taskpool, task->locals);

    /**
     * @brief Entry NULL imples that this task has never been migrated
     * till now in any of the iteration. So we start a new entry.
     */
    if (NULL == (item = parsec_hash_table_nolock_find(task_mapping_ht, key)))
    {

        item = (task_mapping_item_t *)malloc(sizeof(task_mapping_item_t));
        item->device_index = device_index;
        item->ht_item.key = key;
        parsec_hash_table_lock_bucket(task_mapping_ht, key);
        parsec_hash_table_nolock_insert(task_mapping_ht, &item->ht_item);
        parsec_hash_table_unlock_bucket(task_mapping_ht, key);
    }
    else
        item->ht_item.key = key;
}

/**
 * @brief Check if the task has any particular task mapping,
 * if it has return the device_index, or else return -1.
 *
 * @param task
 * @return int
 */
int find_task_to_device_mapping(parsec_task_t *task)
{
    parsec_key_t key;
    task_mapping_item_t *item;

    key = task->task_class->make_key(task->taskpool, task->locals);
    if (NULL == (item = parsec_hash_table_nolock_find(task_mapping_ht, key)))
        return -1;

    return item->device_index;
}

void clear_task_migrated_per_tp()
{
    task_migrated_per_tp = 0;
}

void print_task_migrated_per_tp()
{
    if(parsec_migrate_statistics)
    {
        printf("\n*********** TASKPOOL %d *********** \n", tp_count++);
        printf("Tasks migrated in this TP : %d \n", task_migrated_per_tp);
    }
}
