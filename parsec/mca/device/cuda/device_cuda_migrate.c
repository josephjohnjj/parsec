#include "parsec/mca/device/cuda/device_cuda_migrate.h"
#include "parsec/include/parsec/os-spec-timing.h"
#include "parsec/class/list.h"

extern int parsec_device_cuda_enabled;
extern int parsec_migrate_statistics;
extern int parsec_cuda_migrate_chunk_size; // chunks of task migrated to a device (default=5)
extern int parsec_cuda_migrate_task_selection; // method of task selection (default == single_pass_selection)

parsec_device_cuda_info_t *device_info;
static parsec_list_t *migrated_task_list;           // list of all migrated task
static int NDEVICES;                                // total number of GPUs
static parsec_hash_table_t *task_mapping_ht = NULL; // hashtable for storing task mapping
static int task_migrated_per_tp = 0;
static int tp_count;

static parsec_time_t start;

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
                                            "first_queue_time{double};select_time{double};second_queue_time{double};exec_time_start{double};exec_time_end{double};first_stage_in_time_start{double};sec_stage_in_time_start{double};first_stage_in_time_end{double};sec_stage_in_time_end{double};stage_out_time_start{double};stage_out_time_end{double};complete_time{double};device_index{double};task_count{double};first_waiting_tasks{double};sec_waiting_tasks{double};mig_status{double};nb_first_stage_in{double};nb_sec_stage_in{double};nb_first_stage_in_d2d{double};nb_first_stage_in_h2d{double};nb_sec_stage_in_d2d{double};nb_sec_stage_in_h2d{double};clock_speed{double};task_type{double};class_id{double};exec_stream_index{double}",
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

    start = take_time();

    NDEVICES = ndevices;
    device_info = (parsec_device_cuda_info_t *)calloc(ndevices, sizeof(parsec_device_cuda_info_t));
    migrated_task_list = PARSEC_OBJ_NEW(parsec_list_t);

    for (i = 0; i < NDEVICES; i++)
    {
        for (j = 0; j < EXECUTION_LEVEL; j++)
            device_info[i].task_count[j] = 0;
        device_info[i].load = 0;
        device_info[i].level0 = 0;
        device_info[i].level1 = 0;
        device_info[i].level2 = 0;
        device_info[i].total_tasks_executed = 0;
        device_info[i].received = 0;
        device_info[i].last_device = i;
        device_info[i].deal_count = 0;
        device_info[i].success_count = 0;
        device_info[i].ready_compute_tasks = 0;
        device_info[i].affinity_count = 0;
    }

    task_mapping_ht = PARSEC_OBJ_NEW(parsec_hash_table_t);
    parsec_hash_table_init(task_mapping_ht, offsetof(task_mapping_item_t, ht_item), 16, task_mapping_table_generic_key_fn, NULL);

#if defined(PARSEC_PROF_TRACE)
    gpu_dev_profiling_init();
#endif

#if defined (PARSEC_PROF_TRACE)
    nvmlInit_v2();
#endif

    return 0;
}

int parsec_cuda_migrate_fini()
{
    int i = 0;
    int tot_task_migrated = 0;
    float avg_task_migrated = 0, deal_success_perc = 0, avg_task_migrated_per_sucess;
    int summary_total_tasks_executed = 0, summary_total_compute_tasks_executed = 0;
    int summary_total_tasks_migrated = 0, summary_total_l0_tasks_migrated = 0, summary_total_l1_tasks_migrated = 0, summary_total_l2_tasks_migrated = 0;
    int summary_deals = 0, summary_successful_deals = 0, summary_affinity = 0;
    float summary_avg_task_migrated = 0, summary_deal_success_perc = 0, summary_avg_task_migrated_per_sucess = 0;

#if defined(PARSEC_PROF_TRACE)
    nvmlShutdown();
#endif

    parsec_hash_table_for_all(task_mapping_ht, task_mapping_ht_free_elt, task_mapping_ht);
    parsec_hash_table_fini(task_mapping_ht);
    PARSEC_OBJ_RELEASE(task_mapping_ht);
    task_mapping_ht = NULL;

    if (parsec_migrate_statistics)
    {
        for (i = 0; i < NDEVICES; i++)
        {
            tot_task_migrated = device_info[i].level0 + device_info[i].level1 + device_info[i].level2;
            summary_total_tasks_migrated += tot_task_migrated;
            summary_total_l0_tasks_migrated += device_info[i].level0;
            summary_total_l1_tasks_migrated += device_info[i].level1;
            summary_total_l2_tasks_migrated += device_info[i].level2;
            avg_task_migrated = ((float)tot_task_migrated) / ((float)device_info[i].deal_count);
            deal_success_perc = (((float)device_info[i].success_count) / ((float)device_info[i].deal_count)) * 100;
            avg_task_migrated_per_sucess = ((float)tot_task_migrated) / ((float)device_info[i].success_count);

            printf("\n       *********** DEVICE %d *********** \n", i);
            printf("Total tasks executed                   : %d \n", device_info[i].total_tasks_executed);
            summary_total_tasks_executed += device_info[i].total_tasks_executed;
            printf("Total compute tasks executed           : %d \n", device_info[i].total_compute_tasks);
            printf("Perc of compute tasks                  : %lf \n", ((float)device_info[i].total_compute_tasks / device_info[i].total_tasks_executed) * 100);
            summary_total_compute_tasks_executed += device_info[i].total_compute_tasks;
            printf("Tasks migrated                         : level0 %d, level1 %d, level2 %d (Total %d)\n",
                   device_info[i].level0, device_info[i].level1, device_info[i].level2,
                   tot_task_migrated);
            printf("Tasks with affinity migrated           : %d \n", device_info[i].affinity_count);
            printf("Perc of affinity tasks                 : %lf \n", ((float)device_info[i].affinity_count / tot_task_migrated) * 100);
            summary_affinity += device_info[i].affinity_count;
            printf("Task received                          : %d \n", device_info[i].received);
            printf("Chunk Size                             : %d \n", parsec_cuda_migrate_chunk_size);
            printf("Total deals                            : %d \n", device_info[i].deal_count);
            summary_deals += device_info[i].deal_count;
            printf("Successful deals                       : %d \n", device_info[i].success_count);
            summary_successful_deals += device_info[i].success_count;
            printf("Avg task migrated per deal             : %lf \n", avg_task_migrated);
            printf("Avg task migrated per successfull deal : %lf \n", avg_task_migrated_per_sucess);
            printf("Perc of successfull deals              : %lf \n", deal_success_perc);
        }

        printf("\n      *********** SUMMARY *********** \n");
        printf("Total tasks executed                   : %d \n", summary_total_tasks_executed);
        printf("Total compute tasks executed           : %d \n", summary_total_compute_tasks_executed);
        printf("Perc of compute tasks                  : %lf \n", ((float)summary_total_compute_tasks_executed / summary_total_tasks_executed) * 100);
        printf("Tasks migrated                         : level0 %d, level1 %d, level2 %d (Total %d)\n",
               summary_total_l0_tasks_migrated, summary_total_l1_tasks_migrated, summary_total_l2_tasks_migrated,
               summary_total_tasks_migrated);
        printf("Tasks with affinity migrated           : %d \n", summary_affinity);
        printf("Perc of affinity tasks                 : %lf \n", ((float)summary_affinity / summary_total_tasks_migrated) * 100);
        printf("Total deals                            : %d \n", summary_deals);
        printf("Successful deals                       : %d \n", summary_successful_deals);

        summary_avg_task_migrated = ((float)summary_total_tasks_migrated) / ((float)summary_deals);
        summary_avg_task_migrated_per_sucess = ((float)summary_total_tasks_migrated) / ((float)summary_successful_deals);
        summary_deal_success_perc = (((float)summary_successful_deals) / ((float)summary_deals)) * 100;

        printf("Avg task migrated per deal             : %lf \n", summary_avg_task_migrated);
        printf("Avg task migrated per successfull deal : %lf \n", summary_avg_task_migrated_per_sucess);
        printf("perc of successfull deals              : %lf \n", summary_deal_success_perc);
    }

    if(parsec_cuda_migrate_task_selection == 0)
        printf("Task selection                         : single-try \n" );
    else if(parsec_cuda_migrate_task_selection == 1)
        printf("Task selection                         : single-pass \n" );
    else if(parsec_cuda_migrate_task_selection == 2)
        printf("Task selection                         : two-pass \n" );
    else
        printf("Task selection                         : affinity-only \n" );

     printf("\n---------Execution time = %ld ns ( %lf s)------------ \n", time_stamp(), (double) time_stamp() / 1000000000);
    PARSEC_OBJ_RELEASE(migrated_task_list);
    free(device_info);

    return 0;
}

uint64_t time_stamp()
{
    parsec_time_t now;
    now = take_time();
    return diff_time(start, now);
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
    parsec_device_gpu_module_t *d = (parsec_device_gpu_module_t *) parsec_mca_device_get(DEVICE_NUM(device));
    return (d->mutex < 5) ? 1 : 0;

    // return (parsec_cuda_get_device_task(device, -1) < 5) ? 1 : 0;
    // return (get_compute_tasks_executed(device) < 5) ? 1 : 0;
}

int will_starve(int device)
{
    /**
     * @brief The default number of execution stream in PaRSEC is 2. We assume
     * starvtion if migrating a task will push the number of ready tasks available
     * to less than twice the number of execution stream.
     */
    // parsec_device_gpu_module_t* d = parsec_mca_device_get( DEVICE_NUM(device) );
    // return (d->mutex < 5) ? 1 : 0;

    // return ((parsec_cuda_get_device_task(device, -1) - 1) < 5) ? 1 : 0;
    return (get_compute_tasks_executed(device) < 5) ? 1 : 0;
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
 * @brief Select the victim task for migration.
 *
 * @param es
 * @param ring
 * @param dealer_device
 * @param starving_device
 * @return int
 */

int select_tasks(parsec_execution_stream_t *es, parsec_list_t *ring,
                 parsec_device_gpu_module_t *dealer_device,
                 parsec_device_gpu_module_t *starving_device,
                 int selection_type)
{
    int dealer_device_index = 0;
    int execution_level = 0;
    int deal_success = 0, device_affinity = 0;
    int i = 0;
    parsec_gpu_task_t *migrated_gpu_task = NULL;

    dealer_device_index = CUDA_DEVICE_NUM(dealer_device->super.device_index);

    for (i = 0; i < parsec_cuda_migrate_chunk_size; i++)
    {
        migrated_gpu_task = NULL;

        if (selection_type == 0)
            execution_level = single_try_selection(es, dealer_device, &migrated_gpu_task);
        else if (selection_type == 1) //default
            execution_level = single_pass_selection(es, dealer_device, starving_device, &migrated_gpu_task);
        else if (selection_type == 2)
            execution_level = two_pass_selection(es, dealer_device, starving_device, &migrated_gpu_task);
        else if (selection_type == 3)
            execution_level = affinity_only_selection(es, dealer_device, starving_device, &migrated_gpu_task);

        if (migrated_gpu_task != NULL)
        {
            assert(migrated_gpu_task->ec != NULL);
            PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t *)migrated_gpu_task);

            // keep track of compute task count. Decrement compute task count.
            dec_compute_task_count(dealer_device_index);

            if (parsec_migrate_statistics)
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

            deal_success++;
            parsec_atomic_fetch_inc_int32(&task_migrated_per_tp);

            if (parsec_migrate_statistics)
            {
                if (execution_level == 2)
                    device_affinity = find_task_affinity(migrated_gpu_task, starving_device->super.device_index, TASK_MIGRATED_AFTER_STAGE_IN);
                else
                    device_affinity = find_task_affinity(migrated_gpu_task, starving_device->super.device_index, TASK_MIGRATED_BEFORE_STAGE_IN);

                if (device_affinity)
                    device_info[dealer_device_index].affinity_count++;
            }

            /**
             * @brief change migrate_status according to the status of the stage in of the
             * stage_in data.
             */
            if (execution_level == 2)
                migrated_gpu_task->migrate_status = TASK_MIGRATED_AFTER_STAGE_IN;
            else
                migrated_gpu_task->migrate_status = TASK_MIGRATED_BEFORE_STAGE_IN;

            parsec_list_push_front(ring, (parsec_list_item_t *) migrated_gpu_task);
        }
    } // end for i

    return deal_success;
}

/**
 * @brief Select task  from the different device queues using a single try on each device queue.
 * If first try in a queue fails (that is if the first task is not a compute task or a task that
 * is already migrated) we move on to the next queue.
 *
 * @param es
 * @param dealer_device
 * @param migrated_gpu_task
 * @return int
 */

int single_try_selection(parsec_execution_stream_t *es, parsec_device_gpu_module_t *dealer_device,
                         parsec_gpu_task_t **migrated_gpu_task)
{
    (void)es;
    int j = 0;
    int execution_level = 0;
    *migrated_gpu_task = (parsec_gpu_task_t *)parsec_list_pop_back(&(dealer_device->pending)); // level 0

    if (*migrated_gpu_task != NULL)
    {
        /**
         * @brief if the task is a not a computational kerenel or if it is a task that has
         * already been migrated, we stop the migration and push it back to the queue.
         */
        if ((*migrated_gpu_task)->task_type != PARSEC_GPU_TASK_TYPE_KERNEL || (*migrated_gpu_task)->migrate_status > TASK_NOT_MIGRATED )
        {
            parsec_list_push_back(&(dealer_device->pending), (parsec_list_item_t *) *migrated_gpu_task);
            *migrated_gpu_task = NULL;
        }

        execution_level = 0;
    }

    if (*migrated_gpu_task == NULL)
    {
        // level1 - task is aavailble in the stage_in queue. Stage_in not started.
        *migrated_gpu_task = (parsec_gpu_task_t *)parsec_list_pop_back(dealer_device->exec_stream[0]->fifo_pending); // level 1

        if (*migrated_gpu_task != NULL)
        {
            if ((*migrated_gpu_task)->task_type != PARSEC_GPU_TASK_TYPE_KERNEL || (*migrated_gpu_task)->migrate_status > TASK_NOT_MIGRATED)
            {
                parsec_list_push_back(dealer_device->exec_stream[0]->fifo_pending, (parsec_list_item_t *) *migrated_gpu_task);
                *migrated_gpu_task = NULL;
            }
        }
        execution_level = 1;

        if (*migrated_gpu_task == NULL)
        {
            for (j = 0; j < (dealer_device->num_exec_streams - 2); j++)
            {
                // level2 - task is available in one of the execution queue stage_in is complete
                *migrated_gpu_task = (parsec_gpu_task_t *)parsec_list_pop_back(dealer_device->exec_stream[(2 + j)]->fifo_pending); // level2

                if (*migrated_gpu_task != NULL)
                {
                    if ((*migrated_gpu_task)->task_type != PARSEC_GPU_TASK_TYPE_KERNEL || (*migrated_gpu_task)->migrate_status > TASK_NOT_MIGRATED )
                    {
                        parsec_list_push_back(dealer_device->exec_stream[(2 + j)]->fifo_pending, (parsec_list_item_t *) *migrated_gpu_task);
                        *migrated_gpu_task = NULL;
                    }
                }

                if (*migrated_gpu_task != NULL)
                {
                    execution_level = 2;
                    break;
                }
            }
        } // end of j
    }

    return execution_level;
}

parsec_list_item_t* find_compute_tasks(parsec_list_t *list, parsec_device_gpu_module_t *starving_device, int stage_in_status, 
                                       int pass_count, int selection_type)
{
    parsec_list_item_t *item = NULL;
    parsec_gpu_task_t *task = NULL;
    int device_affinity;

    assert(list != NULL);

    parsec_list_lock(list);

    if ( (pass_count == SECOND_PASS) || (selection_type == SINGLE_PASS_SELECTION) )
    {
        for (item = PARSEC_LIST_ITERATOR_FIRST(list); PARSEC_LIST_ITERATOR_END(list) != item; item = PARSEC_LIST_ITERATOR_NEXT(item))
        {
            task = (parsec_gpu_task_t *)item;

            if ((task->task_type == PARSEC_GPU_TASK_TYPE_KERNEL) && (task->migrate_status == TASK_NOT_MIGRATED))
                break;
        }
    }
    else if ( (pass_count == FIRST_PASS) || (selection_type == AFFINITY_ONLY_SELECTION) )
    {
        for (item = PARSEC_LIST_ITERATOR_FIRST(list); PARSEC_LIST_ITERATOR_END(list) != item; item = PARSEC_LIST_ITERATOR_NEXT(item))
        {
            task = (parsec_gpu_task_t *)item;
            device_affinity = find_task_affinity(task, starving_device->super.device_index, stage_in_status);

            if ((task->task_type == PARSEC_GPU_TASK_TYPE_KERNEL) && (task->migrate_status == TASK_NOT_MIGRATED) && (device_affinity > 0))
                break;
        }
    }

    parsec_list_unlock(list);

    if ((item != NULL) && (PARSEC_LIST_ITERATOR_END(list) != item))
    {
        parsec_list_nolock_remove(list, item);
        return item;
    }

    return NULL;
}

/**
 * @brief Select task  from the different device queues using a single pass through the
 * device queues. 
 * @param es
 * @param dealer_device
 * @param migrated_gpu_task
 * @return int
 */

int single_pass_selection(parsec_execution_stream_t *es, parsec_device_gpu_module_t *dealer_device,
                          parsec_device_gpu_module_t *starving_device, parsec_gpu_task_t **migrated_gpu_task)
{
    int j = 0;
    int execution_level = 0;

    *migrated_gpu_task = (parsec_gpu_task_t *)find_compute_tasks(&(dealer_device->pending), starving_device, 
                                            TASK_MIGRATED_BEFORE_STAGE_IN, -1, SINGLE_PASS_SELECTION);
    execution_level = 0;

    if (*migrated_gpu_task == NULL)
    {
        // level1 - task is availble in the stage_in queue. Stage_in not started.
        *migrated_gpu_task = (parsec_gpu_task_t *)find_compute_tasks(dealer_device->exec_stream[0]->fifo_pending, starving_device, 
                                            TASK_MIGRATED_BEFORE_STAGE_IN, -1, SINGLE_PASS_SELECTION);
        execution_level = 1;

        if (*migrated_gpu_task == NULL)
        {
            for (j = 0; j < (dealer_device->num_exec_streams - 2); j++)
            {
                // level2 - task is available in one of the execution queue stage_in is complete
                *migrated_gpu_task = (parsec_gpu_task_t *)find_compute_tasks(dealer_device->exec_stream[(2 + j)]->fifo_pending, starving_device, 
                                            TASK_MIGRATED_AFTER_STAGE_IN, -1, SINGLE_PASS_SELECTION);

                if (*migrated_gpu_task != NULL)
                {
                    execution_level = 2;
                    break;
                }
            }
        } // end of j
    }

    (void)es;
    return execution_level;
}

/**
 * @brief Select task  from the different device queues using a two pass through the
 * device queues. The first pass only selects a task with an affinity to the starving
 * device. If the first pass does not yield any tasks, the second pass selects any available 
 * compute tasks.
 *
 * @param es
 * @param dealer_device
 * @param migrated_gpu_task
 * @return int
 */
int two_pass_selection(parsec_execution_stream_t *es, parsec_device_gpu_module_t *dealer_device,
                       parsec_device_gpu_module_t *starving_device, parsec_gpu_task_t **migrated_gpu_task)
{
    int j = 0;
    int execution_level = 0;

    // FIRST PASS

    *migrated_gpu_task = (parsec_gpu_task_t *)find_compute_tasks(&(dealer_device->pending), starving_device, 
                                TASK_MIGRATED_BEFORE_STAGE_IN, FIRST_PASS, TWO_PASS_SELECTION);
    execution_level = 0;

    if (*migrated_gpu_task == NULL)
    {
        // level1 - task is availble in the stage_in queue. Stage_in not started.
        *migrated_gpu_task = (parsec_gpu_task_t *)find_compute_tasks(dealer_device->exec_stream[0]->fifo_pending, starving_device, 
                                TASK_MIGRATED_BEFORE_STAGE_IN, FIRST_PASS, TWO_PASS_SELECTION);
        execution_level = 1;

        if (*migrated_gpu_task == NULL)
        {
            for (j = 0; j < (dealer_device->num_exec_streams - 2); j++)
            {
                // level2 - task is available in one of the execution queue stage_in is complete
                *migrated_gpu_task = (parsec_gpu_task_t *)find_compute_tasks(dealer_device->exec_stream[(2 + j)]->fifo_pending, starving_device, 
                                TASK_MIGRATED_AFTER_STAGE_IN, FIRST_PASS, TWO_PASS_SELECTION);

                if (*migrated_gpu_task != NULL)
                {
                    execution_level = 2;
                    break;
                }
            }
        } // end of j
    }

    // SECOND PASS

    if (*migrated_gpu_task == NULL)
    {
        *migrated_gpu_task = (parsec_gpu_task_t *)find_compute_tasks(&(dealer_device->pending), starving_device, 
                                TASK_MIGRATED_BEFORE_STAGE_IN, SECOND_PASS, TWO_PASS_SELECTION);
        execution_level = 0;

        if (*migrated_gpu_task == NULL)
        {
            // level1 - task is availble in the stage_in queue. Stage_in not started.
            *migrated_gpu_task = (parsec_gpu_task_t *)find_compute_tasks(dealer_device->exec_stream[0]->fifo_pending, starving_device, 
                                    TASK_MIGRATED_BEFORE_STAGE_IN, SECOND_PASS, TWO_PASS_SELECTION);
            execution_level = 1;

            if (*migrated_gpu_task == NULL)
            {
                for (j = 0; j < (dealer_device->num_exec_streams - 2); j++)
                {
                    // level2 - task is available in one of the execution queue stage_in is complete
                    *migrated_gpu_task = (parsec_gpu_task_t *)find_compute_tasks(dealer_device->exec_stream[(2 + j)]->fifo_pending, starving_device, 
                                            TASK_MIGRATED_AFTER_STAGE_IN, SECOND_PASS, TWO_PASS_SELECTION);

                    if (*migrated_gpu_task != NULL)
                    {
                        execution_level = 2;
                        break;
                    }
                }
            } // end of j
        }
    }

    (void )es;

    return execution_level;
}


/**
 * @brief Select task  from the different device queues using a single pass through the
 * device queues. 
 * @param es
 * @param dealer_device
 * @param migrated_gpu_task
 * @return int
 */

int affinity_only_selection(parsec_execution_stream_t *es, parsec_device_gpu_module_t *dealer_device,
                          parsec_device_gpu_module_t *starving_device, parsec_gpu_task_t **migrated_gpu_task)
{
    int j = 0;
    int execution_level = 0;

    *migrated_gpu_task = (parsec_gpu_task_t *)find_compute_tasks(&(dealer_device->pending), starving_device, 
                                            TASK_MIGRATED_BEFORE_STAGE_IN, -1, AFFINITY_ONLY_SELECTION);
    execution_level = 0;

    if (*migrated_gpu_task == NULL)
    {
        // level1 - task is availble in the stage_in queue. Stage_in not started.
        *migrated_gpu_task = (parsec_gpu_task_t *)find_compute_tasks(dealer_device->exec_stream[0]->fifo_pending, starving_device, 
                                            TASK_MIGRATED_BEFORE_STAGE_IN, -1, AFFINITY_ONLY_SELECTION);
        execution_level = 1;

        if (*migrated_gpu_task == NULL)
        {
            for (j = 0; j < (dealer_device->num_exec_streams - 2); j++)
            {
                // level2 - task is available in one of the execution queue stage_in is complete
                *migrated_gpu_task = (parsec_gpu_task_t *)find_compute_tasks(dealer_device->exec_stream[(2 + j)]->fifo_pending, starving_device, 
                                            TASK_MIGRATED_AFTER_STAGE_IN, -1, AFFINITY_ONLY_SELECTION);

                if (*migrated_gpu_task != NULL)
                {
                    execution_level = 2;
                    break;
                }
            }
        } // end of j
    }

    (void)es;
    return execution_level;
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
    int nb_migrated = 0, execution_level = 0;
    int deal_success = 0;
    int i = 0, k = 0, d = 0;
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
        device_info[dealer_device_index].deal_count++;

        parsec_list_t *ring = PARSEC_OBJ_NEW(parsec_list_t);
        PARSEC_OBJ_RETAIN(ring);
        deal_success = select_tasks(es, ring, dealer_device, starving_device, parsec_cuda_migrate_task_selection);
        nb_migrated += deal_success;

        while (!parsec_list_nolock_is_empty(ring))
        {
            migrated_gpu_task = (parsec_gpu_task_t *) parsec_list_pop_front(ring);
            assert(migrated_gpu_task != NULL);

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
            for (k = 0; k < MAX_PARAM_COUNT; k++)
                migrated_gpu_task->candidate[i] = NULL;
            mig_task->dealer_device = dealer_device;
            mig_task->starving_device = starving_device;
            mig_task->stage_in_status = migrated_gpu_task->migrate_status;
#if defined(PARSEC_PROF_TRACE)
            migrated_gpu_task->select_time = time_stamp();
#endif

            PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t *)mig_task);
            parsec_cuda_mig_task_enqueue(es, mig_task);

            device_info[dealer_device_index].last_device = starving_device_index;
            char tmp[MAX_TASK_STRLEN];
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream, "Task %s migrated (level %d, stage_in %d) from device %d to device %d",
                                 parsec_task_snprintf(tmp, MAX_TASK_STRLEN, ((parsec_gpu_task_t *)migrated_gpu_task)->ec),
                                 execution_level, mig_task->stage_in_status, dealer_device_index, starving_device_index);
        } // end while

        if (deal_success > 0)
            device_info[dealer_device_index].success_count++;

        if (will_starve(dealer_device_index))
            break;
    } // end for d

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

        return 1;
    }
    else
        item->ht_item.key = key;

    return 0;
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
    if (parsec_migrate_statistics)
    {
        printf("\n*********** TASKPOOL %d *********** \n", tp_count++);
        printf("Tasks migrated in this TP : %d \n", task_migrated_per_tp);
    }
}

int inc_compute_task_count(int device_index)
{
    parsec_atomic_fetch_inc_int32(&device_info[device_index].ready_compute_tasks);
    return device_info[device_index].ready_compute_tasks;
}

int dec_compute_task_count(int device_index)
{
    parsec_atomic_fetch_dec_int32(&device_info[device_index].ready_compute_tasks);
    return device_info[device_index].ready_compute_tasks;
}

int inc_compute_tasks_executed(int device_index)
{
    parsec_atomic_fetch_inc_int32(&device_info[device_index].total_compute_tasks);
    return device_info[device_index].total_compute_tasks;
}

int get_compute_tasks_executed(int device_index)
{
    return device_info[device_index].total_compute_tasks;
}

int find_task_affinity(parsec_gpu_task_t *gpu_task, int device_index, int status)
{
    int i;
    parsec_data_t *original = NULL;
    parsec_data_copy_t *data_copy = NULL;
    parsec_task_t *this_task = gpu_task->ec;

    for (i = 0; i < this_task->task_class->nb_flows; i++)
    {
        if (NULL == this_task->data[i].data_in)
            continue;
        if (NULL == this_task->data[i].source_repo_entry)
            continue;

        if (status == TASK_MIGRATED_BEFORE_STAGE_IN) // data will be trasfered from data_in
        {
            original = this_task->data[i].data_in->original;
            data_copy = this_task->data[i].data_in;
        }
        else // data will be trasfered from data_out
        {
            original = this_task->data[i].data_out->original;
            data_copy = this_task->data[i].data_out;
        }

        if (original->device_copies[device_index] != NULL &&
            data_copy->version == original->device_copies[device_index]->version)

        {
            /**
             * If both the both the data copy has the same version, there is no need
             * for a data transfer.
             */
            return 1;
        }
    }

    return 0;
}
