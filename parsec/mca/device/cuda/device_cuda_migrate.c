#include "parsec/mca/device/cuda/device_cuda_migrate.h"
#include "parsec/include/parsec/os-spec-timing.h"
#include "parsec/class/list.h"
#include "parsec/execution_stream.h"

extern int parsec_device_cuda_enabled;
extern int parsec_migrate_statistics;
extern int parsec_cuda_migrate_tasks;
extern int parsec_cuda_migrate_chunk_size;       // chunks of task migrated to a device (default=5)
extern int parsec_cuda_migrate_task_selection;   // method of task selection (default == single_pass_selection)
extern int parsec_cuda_delegate_task_completion; // task completion delegation
extern int parsec_cuda_iterative;

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
                                            "first_queue_time{double};select_time{double};second_queue_time{double};"
                                            "exec_time_start{double};exec_time_end{double};"
                                            "first_stage_in_time_start{double};sec_stage_in_time_start{double};"
                                            "first_stage_in_time_end{double};sec_stage_in_time_end{double};"
                                            "stage_out_time_start{double};stage_out_time_end{double};"
                                            "complete_time_start{double};complete_time_end{double};"
                                            "device_index{double};task_count{double};"
                                            "first_waiting_tasks{double};sec_waiting_tasks{double};"
                                            "mig_status{double};nb_first_stage_in{double};nb_sec_stage_in{double};"
                                            "nb_first_stage_in_d2d{double};nb_first_stage_in_h2d{double};"
                                            "nb_sec_stage_in_d2d{double};nb_sec_stage_in_h2d{double};clock_speed{double};"
                                            "task_type{double};class_id{double};exec_stream_index{double}",
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
        device_info[i].load                 = 0;
        device_info[i].level0               = 0;
        device_info[i].level1               = 0;
        device_info[i].level2               = 0;
        device_info[i].total_tasks_executed = 0;
        device_info[i].received             = 0;
        device_info[i].last_device          = i;
        device_info[i].deal_count           = 0;
        device_info[i].success_count        = 0;
        device_info[i].ready_compute_tasks  = 0;
        device_info[i].affinity_count       = 0;
        device_info[i].evictions            = 0;
        device_info[i].nb_stage_in          = 0;
        device_info[i].nb_stage_in_req      = 0;
        device_info[i].completed_co_manager = 0;
        device_info[i].completed_manager    = 0;
        device_info[i].iterative_mapped     = 0;
        device_info[i].thrashing            = 0;
    }

    task_mapping_ht = PARSEC_OBJ_NEW(parsec_hash_table_t);
    parsec_hash_table_init(task_mapping_ht, offsetof(task_mapping_item_t, ht_item), 16, task_mapping_table_generic_key_fn, NULL);

#if defined(PARSEC_PROF_TRACE)
    gpu_dev_profiling_init();
#endif

#if defined(PARSEC_PROF_TRACE)
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
    int summary_total_evictions = 0, summary_total_stage_in = 0, summary_total_stage_in_req = 0;
    int summary_completed_manager = 0, summary_completed_co_manager  = 0;
    int summary_iterative_mapped = 0, summary_thrashing = 0;

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
            summary_total_evictions += device_info[i].evictions;
            summary_total_stage_in += device_info[i].nb_stage_in;
            summary_total_stage_in_req += device_info[i].nb_stage_in_req;
            summary_completed_manager += device_info[i].completed_manager;
            summary_completed_co_manager += device_info[i].completed_co_manager;
            summary_iterative_mapped += device_info[i].iterative_mapped;
            summary_thrashing += device_info[i].thrashing; 

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
            printf("Evictions                              : %d \n", device_info[i].evictions);
            printf("Thrashing                              : %d \n", device_info[i].thrashing);
            printf("Stage in initiated                     : %d \n", device_info[i].nb_stage_in);
            printf("Stage in required                      : %d \n", device_info[i].nb_stage_in_req);
            printf("Perc eviction for stage in initiated   : %lf \n", (( (float)device_info[i].evictions / device_info[i].nb_stage_in) * 100 ) );
            printf("Perc eviction for stage in required    : %lf \n", (((float)device_info[i].evictions / device_info[i].nb_stage_in_req) * 100 ));
            printf("Tasks completed by manager             : %d  \n", device_info[i].completed_manager);
            printf("Tasks completed by co-manager          : %d  \n", device_info[i].completed_co_manager);
            printf("Tasks mapped iteratively               : %d  \n", device_info[i].iterative_mapped);
        }

        printf("\n      *********** SUMMARY *********** \n");
        printf("Total devices                          : %d \n", NDEVICES);
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

        printf("Total evictions                        : %d \n", summary_total_evictions);
        printf("Total Thrashing                        : %d \n", summary_thrashing);
        printf("Total stage in initiated               : %d \n", summary_total_stage_in);
        printf("Total stage in required                : %d \n", summary_total_stage_in_req);
        printf("Perc eviction for stage in initiated   : %lf \n", (((float)summary_total_evictions / summary_total_stage_in) * 100 ) );
        printf("Perc eviction for stage in required    : %lf \n", (((float)summary_total_evictions / summary_total_stage_in_req) * 100 ) );

        printf("Tasks completed by manager             : %d  \n", summary_completed_manager);
        printf("Tasks completed by co-manager          : %d  \n", summary_completed_co_manager);
        printf("Tasks mapped iteratively               : %d  \n", summary_iterative_mapped);
        

        

        if (parsec_cuda_migrate_task_selection == SINGLE_TRY_SELECTION)
            printf("Task selection                         : single-try \n");
        else if (parsec_cuda_migrate_task_selection == SINGLE_PASS_SELECTION)
            printf("Task selection                         : single-pass \n");
        else if (parsec_cuda_migrate_task_selection == TWO_PASS_SELECTION)
            printf("Task selection                         : two-pass \n");
        else if (parsec_cuda_migrate_task_selection == AFFINITY_ONLY_SELECTION)
            printf("Task selection                         : affinity-only \n");
        else if (parsec_cuda_migrate_task_selection == DATA_REUSE_SELECTION)
            printf("Task selection                         : data-reuse \n");

        if (parsec_cuda_delegate_task_completion == 0)
            printf("Task completion                        : not delegated\n");
        else
            printf("Task completion                        : delegated\n");

        if ( parsec_cuda_migrate_tasks == 0)
            printf("Migration                              : no migration \n");
        else if ( parsec_cuda_migrate_tasks == 1)
            printf("Migration                              : not delegated \n");
        else
            printf("Migration                              : delegated \n");

        if(parsec_cuda_iterative == 1)
            printf("Iterative task mapping                 : mig task mapped \n");
        else if(parsec_cuda_iterative == 2)
            printf("Iterative task mapping                 : all tasks mapped \n");
        else
            printf("Iterative task mapping                 : no \n");


        printf("\n---------Execution time = %ld ns ( %lf s)------------ \n", time_stamp(), (double)time_stamp() / 1000000000);
    }

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

int parsec_cuda_inc_eviction_count(int device)
{
    int rc = parsec_atomic_fetch_add_int32(&(device_info[device].evictions), 1);
    return rc + 1;
}

int parsec_cuda_inc_stage_in_count(int device)
{
    int rc = parsec_atomic_fetch_add_int32(&(device_info[device].nb_stage_in), 1);
    return rc + 1;
}

int parsec_cuda_inc_stage_in_req_count(int device)
{
    int rc = parsec_atomic_fetch_add_int32(&(device_info[device].nb_stage_in_req), 1);
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
    parsec_device_gpu_module_t *d = (parsec_device_gpu_module_t *)parsec_mca_device_get(DEVICE_NUM(device));
    return (d->mutex < d->num_exec_streams ) ? 1 : 0;
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

int set_migrate_status(parsec_device_gpu_module_t *dealer_device, parsec_device_gpu_module_t *starving_device,
                       parsec_gpu_task_t *migrated_gpu_task, int execution_level)
{
    int dealer_device_index = CUDA_DEVICE_NUM(dealer_device->super.device_index);
    int affinity = 0;

    /**
     * @brief change migrate_status according to the status of the stage in of the
     * stage_in data.
     */
    if (execution_level == 2)
        migrated_gpu_task->migrate_status = TASK_MIGRATED_AFTER_STAGE_IN;
    else
        migrated_gpu_task->migrate_status = TASK_MIGRATED_BEFORE_STAGE_IN;

    // keep track of compute task count. Decrement compute task count.
    dec_compute_task_count(dealer_device_index);

    if (parsec_migrate_statistics)
    {
        if (execution_level == 0)
        {
            device_info[dealer_device_index].level0++;
        }
        if (execution_level == 1)
        {
            device_info[dealer_device_index].level1++;
        }
        if (execution_level == 2)
        {
            device_info[dealer_device_index].level2++;
        }
    }

    parsec_atomic_fetch_inc_int32(&task_migrated_per_tp);
    if (parsec_migrate_statistics)
    {
        if (execution_level == 2)
            affinity = find_task_affinity_to_starving_node(migrated_gpu_task, starving_device->super.device_index, TASK_MIGRATED_AFTER_STAGE_IN);
        else
            affinity = find_task_affinity_to_starving_node(migrated_gpu_task, starving_device->super.device_index, TASK_MIGRATED_BEFORE_STAGE_IN);
        if (affinity)
            device_info[dealer_device_index].affinity_count++;
    }

    return migrated_gpu_task->migrate_status;
}

/**
 * @brief Select the victim task for migration.
 *
 * @param es
 * @param ring: ring of selected task returned by the selection policy
 * @param dealer_device
 * @param starving_device
 * @param selection_type: mca parameter
 * @return int
 */

int select_tasks(parsec_execution_stream_t *es, parsec_list_t *ring,
                 parsec_device_gpu_module_t *dealer_device, parsec_device_gpu_module_t *starving_device,
                 int selection_type)
{
    int deal_success = 0;

    if (selection_type == SINGLE_TRY_SELECTION)
        deal_success = single_try_selection(es, dealer_device, starving_device, ring);
    else if (selection_type == SINGLE_PASS_SELECTION) // default
        deal_success = single_pass_selection(es, dealer_device, starving_device, ring);
    else if (selection_type == TWO_PASS_SELECTION)
        deal_success = two_pass_selection(es, dealer_device, starving_device, ring);
    else if (selection_type == AFFINITY_ONLY_SELECTION)
        deal_success = affinity_only_selection(es, dealer_device, starving_device, ring);
    else if (selection_type == DATA_REUSE_SELECTION)
        deal_success = data_reuse_selection(es, dealer_device, starving_device, ring);

    return deal_success;
}

int find_compute_tasks(parsec_list_t *list, parsec_device_gpu_module_t *dealer_device,
                       parsec_device_gpu_module_t *starving_device, int stage_in_status,
                       int pass_count, int selection_type, int execution_level, parsec_list_t *ring,
                       int *tries, int *deal_success)
{
    parsec_list_item_t *item = NULL;
    parsec_gpu_task_t *task = NULL;
    int affinity;

    assert(list != NULL);

    parsec_list_lock(list);

    if ( selection_type == SINGLE_TRY_SELECTION )
    {
        do 
        {
            task = (parsec_gpu_task_t *)parsec_list_nolock_pop_back(list);
            if (task != NULL)
            {
                if ((task->task_type == PARSEC_GPU_TASK_TYPE_KERNEL) && (task->migrate_status == TASK_NOT_MIGRATED))
                {
                    set_migrate_status(dealer_device, starving_device, task, execution_level);
                    PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t *)task);
                    parsec_list_nolock_push_back(ring, (parsec_list_item_t *)task);
                    *deal_success += 1;
                    *tries += 1;
                }
                else
                {
                    parsec_list_nolock_push_back(list, (parsec_list_item_t *)task);
                    task = NULL;
                }
            }
        } while ((*tries < parsec_cuda_migrate_chunk_size) && (task != NULL));
    }

    if ( selection_type == DATA_REUSE_SELECTION )
    {
        
       
        for (item = PARSEC_LIST_ITERATOR_FIRST(list);
         (PARSEC_LIST_ITERATOR_END(list) != item) && (*tries < parsec_cuda_migrate_chunk_size);
         item = PARSEC_LIST_ITERATOR_NEXT(item))
        {
            parsec_gpu_task_t *selected_task;
            if(parsec_list_nolock_is_empty( ring ))
                selected_task = NULL;
            else
                selected_task = (parsec_gpu_task_t *)PARSEC_LIST_ITERATOR_LAST(ring); 

            task = (parsec_gpu_task_t *)item;
            affinity = find_task_to_task_affinity(selected_task, task, stage_in_status);  
            if ((task != NULL) && (task->task_type == PARSEC_GPU_TASK_TYPE_KERNEL) &&
                (task->migrate_status == TASK_NOT_MIGRATED) && (affinity > 0))
            {
                item = parsec_list_nolock_remove(list, item);
                set_migrate_status(dealer_device, starving_device, task, execution_level);
                PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t *)task);
                parsec_list_push_back(ring, (parsec_list_item_t *)task);
                *tries = *tries + 1;
                *deal_success += 1;
            }
        }

        

    }

    if (( pass_count == SECOND_PASS && selection_type == TWO_PASS_SELECTION ) || 
        ( selection_type == SINGLE_PASS_SELECTION ))
    {
        for (item = PARSEC_LIST_ITERATOR_FIRST(list);
             (PARSEC_LIST_ITERATOR_END(list) != item) && (*tries < parsec_cuda_migrate_chunk_size);
             item = PARSEC_LIST_ITERATOR_NEXT(item))
        {
            task = (parsec_gpu_task_t *)item;

            if ((task != NULL) && (task->task_type == PARSEC_GPU_TASK_TYPE_KERNEL) && (task->migrate_status == TASK_NOT_MIGRATED))
            {
                /**
                 * parsec_list_nolock_remove returns the previous element in the
                 * linked list. This will preserve the chain of iteration.
                 */
                item = parsec_list_nolock_remove(list, item);
                set_migrate_status(dealer_device, starving_device, task, execution_level);
                PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t *)task);
                parsec_list_push_back(ring, (parsec_list_item_t *)task);
                *tries = *tries + 1;
                *deal_success += 1;
            }
        }
    }
    else if (( pass_count == FIRST_PASS && selection_type == TWO_PASS_SELECTION ) || 
             ( selection_type == AFFINITY_ONLY_SELECTION))
    {
        for (item = PARSEC_LIST_ITERATOR_FIRST(list);
             (PARSEC_LIST_ITERATOR_END(list) != item) && (*tries < parsec_cuda_migrate_chunk_size);
             item = PARSEC_LIST_ITERATOR_NEXT(item))
        {
            task = (parsec_gpu_task_t *)item;
            affinity = find_task_affinity_to_starving_node(task, starving_device->super.device_index, stage_in_status);

            if ((task != NULL) && (task->task_type == PARSEC_GPU_TASK_TYPE_KERNEL) &&
                (task->migrate_status == TASK_NOT_MIGRATED) && (affinity > 0))
            {
                item = parsec_list_nolock_remove(list, item);
                set_migrate_status(dealer_device, starving_device, task, execution_level);
                PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t *)task);
                parsec_list_push_back(ring, (parsec_list_item_t *)task);
                *tries = *tries + 1;
                *deal_success += 1;
            }
        }
    }

    parsec_list_unlock(list);

    return *deal_success;
}

/**
 * @brief Tries to select the first task in a queue. If that fails it moves on to the 
 * next queue.
 */

int single_try_selection(parsec_execution_stream_t *es, parsec_device_gpu_module_t *dealer_device,
                         parsec_device_gpu_module_t *starving_device, parsec_list_t *ring)
{
    int j = 0;
    int execution_level = 0;  
    /**
     * @brief Keep tracks of the number of times we try to select a task.
     * Upper limit is the parsec_cuda_migrate_chunk_size set by the
     * mca parameter
     */
    int tries = 0;
    /**
     * @brief Keeps track of the number of successfull tasks migrated.
     * This is very important as this value is deducted from the total tasks 
     * the dealer device will execute
     * 
     */
    int deal_success = 0;

    (void)es;

    find_compute_tasks(&(dealer_device->pending), dealer_device, starving_device,
                       TASK_MIGRATED_BEFORE_STAGE_IN, -1, SINGLE_TRY_SELECTION, execution_level, ring, &tries, &deal_success);

    if (tries < parsec_cuda_migrate_chunk_size)
    {
        // level1 - task is availble in the stage_in queue. Stage_in not started.
        execution_level = 1;
        find_compute_tasks(dealer_device->exec_stream[0]->fifo_pending, dealer_device, starving_device,
                           TASK_MIGRATED_BEFORE_STAGE_IN, -1, SINGLE_TRY_SELECTION, execution_level, ring, &tries, &deal_success);

        for (j = 0; j < (dealer_device->num_exec_streams - 2); j++)
        {
            if (tries < parsec_cuda_migrate_chunk_size)
            {
                // level2 - task is available in one of the execution queue stage_in is complete
                execution_level = 2;
                find_compute_tasks(dealer_device->exec_stream[(2 + j)]->fifo_pending, dealer_device, starving_device,
                                   TASK_MIGRATED_AFTER_STAGE_IN, -1, SINGLE_TRY_SELECTION, execution_level, ring, &tries, &deal_success);
            }
            else
                break;
        }
    }

    return deal_success;
}

/**
 * @brief Select task  from the different device queues using a single pass through the
 * device queues.
 */

int single_pass_selection(parsec_execution_stream_t *es, parsec_device_gpu_module_t *dealer_device,
                          parsec_device_gpu_module_t *starving_device, parsec_list_t *ring)
{
    int j = 0;
    int execution_level = 0;
    /**
     * @brief Keep tracks of the number of times we try to select a task.
     * Upper limit is the parsec_cuda_migrate_chunk_size set by the
     * mca parameter
     */
    int tries = 0;
    /**
     * @brief Keeps track of the number of successfull tasks migrated.
     * This is very important as this value is deducted from the total tasks 
     * the dealer device will execute
     * 
     */
    int deal_success = 0;

    find_compute_tasks(&(dealer_device->pending), dealer_device, starving_device,
                       TASK_MIGRATED_BEFORE_STAGE_IN, -1, SINGLE_PASS_SELECTION, execution_level, ring, &tries, &deal_success);

    if (tries < parsec_cuda_migrate_chunk_size)
    {
        // level1 - task is availble in the stage_in queue. Stage_in not started.
        execution_level = 1;
        find_compute_tasks(dealer_device->exec_stream[0]->fifo_pending, dealer_device, starving_device,
                           TASK_MIGRATED_BEFORE_STAGE_IN, -1, SINGLE_PASS_SELECTION, execution_level, ring, &tries, &deal_success);

        for (j = 0; j < (dealer_device->num_exec_streams - 2); j++)
        {
            if (tries < parsec_cuda_migrate_chunk_size)
            {
                // level2 - task is available in one of the execution queue stage_in is complete
                execution_level = 2;
                find_compute_tasks(dealer_device->exec_stream[(2 + j)]->fifo_pending, dealer_device, starving_device,
                                   TASK_MIGRATED_AFTER_STAGE_IN, -1, SINGLE_PASS_SELECTION, execution_level, ring, &tries, &deal_success);
            }
            else
                break;
        }

    }

    (void)es;
    return deal_success;
}

/**
 * @brief Select task  from the different device queues using a two pass through the
 * device queues. The first pass only selects a task with an affinity to the starving
 * device. If the first pass does not yield the required number of tasks, the 
 * second pass selects any available compute tasks.
 */
int two_pass_selection(parsec_execution_stream_t *es, parsec_device_gpu_module_t *dealer_device,
                       parsec_device_gpu_module_t *starving_device, parsec_list_t *ring)
{
    int j = 0;
    int execution_level = 0;
    /**
    * @brief Keep tracks of the number of times we try to select a task.
    * Upper limit is the parsec_cuda_migrate_chunk_size set by the
    * mca parameter
    */
    int tries = 0;
    /**
     * @brief Keeps track of the number of successfull tasks migrated.
     * This is very important as this value is deducted from the total tasks 
     * the dealer device will execute
     * 
     */
    int deal_success = 0;

    (void)es;

    // FIRST PASS

    find_compute_tasks(&(dealer_device->pending), dealer_device, starving_device,
                       TASK_MIGRATED_BEFORE_STAGE_IN, FIRST_PASS, TWO_PASS_SELECTION, execution_level, ring, &tries, &deal_success);

    if (tries < parsec_cuda_migrate_chunk_size)
    {
        // level1 - task is availble in the stage_in queue. Stage_in not started.
        execution_level = 1;
        find_compute_tasks(dealer_device->exec_stream[0]->fifo_pending, dealer_device, starving_device,
                           TASK_MIGRATED_BEFORE_STAGE_IN, FIRST_PASS, TWO_PASS_SELECTION, execution_level, ring, &tries, &deal_success);

        for (j = 0; j < (dealer_device->num_exec_streams - 2); j++)
        {
            if (tries < parsec_cuda_migrate_chunk_size)
            {
                // level2 - task is available in one of the execution queue stage_in is complete
                execution_level = 2;
                find_compute_tasks(dealer_device->exec_stream[(2 + j)]->fifo_pending, dealer_device, starving_device,
                                   TASK_MIGRATED_AFTER_STAGE_IN, FIRST_PASS, TWO_PASS_SELECTION, execution_level, ring, &tries, &deal_success);
            }
            else
                break;
        }
    }

    // SECOND PASS

    if (tries < parsec_cuda_migrate_chunk_size)
    {
        execution_level = 0;
        find_compute_tasks(&(dealer_device->pending), dealer_device, starving_device,
                           TASK_MIGRATED_BEFORE_STAGE_IN, SECOND_PASS, TWO_PASS_SELECTION, execution_level, ring, &tries, &deal_success);

        if (tries < parsec_cuda_migrate_chunk_size)
        {
            // level1 - task is availble in the stage_in queue. Stage_in not started.
            execution_level = 1;
            find_compute_tasks(dealer_device->exec_stream[0]->fifo_pending, dealer_device, starving_device,
                               TASK_MIGRATED_BEFORE_STAGE_IN, SECOND_PASS, TWO_PASS_SELECTION, execution_level, ring, &tries, &deal_success);

            for (j = 0; j < (dealer_device->num_exec_streams - 2); j++)
            {
                if (tries < parsec_cuda_migrate_chunk_size)
                {
                    // level2 - task is available in one of the execution queue stage_in is complete
                    execution_level = 2;
                    find_compute_tasks(dealer_device->exec_stream[(2 + j)]->fifo_pending, dealer_device, starving_device,
                                       TASK_MIGRATED_AFTER_STAGE_IN, SECOND_PASS, TWO_PASS_SELECTION, execution_level, ring, &tries, &deal_success);
                }
                else
                    break;
            }
        }
    }

    return deal_success;
}

/**
 * @brief Select only the tasks with an affinity with the starving device.
 */

int affinity_only_selection(parsec_execution_stream_t *es, parsec_device_gpu_module_t *dealer_device,
                            parsec_device_gpu_module_t *starving_device, parsec_list_t *ring)
{
    int j = 0;
    int execution_level = 0;
    /**
    * @brief Keep tracks of the number of times we try to select a task.
    * Upper limit is the parsec_cuda_migrate_chunk_size set by the
    * mca parameter
    */
    int tries = 0;
    /**
     * @brief Keeps track of the number of successfull tasks migrated.
     * This is very important as this value is deducted from the total tasks 
     * the dealer device will execute
     * 
     */
    int deal_success = 0;

    find_compute_tasks(&(dealer_device->pending), dealer_device, starving_device,
                       TASK_MIGRATED_BEFORE_STAGE_IN, -1, AFFINITY_ONLY_SELECTION, execution_level, ring, &tries, &deal_success);

    if (tries < parsec_cuda_migrate_chunk_size)
    {
        // level1 - task is availble in the stage_in queue. Stage_in not started.
        execution_level = 1;
        find_compute_tasks(dealer_device->exec_stream[0]->fifo_pending, dealer_device, starving_device,
                           TASK_MIGRATED_BEFORE_STAGE_IN, -1, AFFINITY_ONLY_SELECTION, execution_level, ring, &tries, &deal_success);

        for (j = 0; j < (dealer_device->num_exec_streams - 2); j++)
        {
            if (tries < parsec_cuda_migrate_chunk_size)
            {
                // level2 - task is available in one of the execution queue stage_in is complete
                execution_level = 2;
                find_compute_tasks(dealer_device->exec_stream[(2 + j)]->fifo_pending, dealer_device, starving_device,
                                   TASK_MIGRATED_AFTER_STAGE_IN, -1, AFFINITY_ONLY_SELECTION, execution_level, ring, &tries, &deal_success);
            }
            else
                break;
        }
    }

    (void)es;
    return deal_success;
}

/**
 * @brief Tries to select tasks that has common data between each other. 
 * This will ensure a degree of data resuse in the starving node.
 */

int data_reuse_selection(parsec_execution_stream_t *es, parsec_device_gpu_module_t *dealer_device,
                         parsec_device_gpu_module_t *starving_device, parsec_list_t *ring)
{
    int j = 0;
    int execution_level = 0;
    /**
    * @brief Keep tracks of the number of times we try to select a task.
    * Upper limit is the parsec_cuda_migrate_chunk_size set by the
    * mca parameter
    */
    int tries = 0;
    /**
     * @brief Keeps track of the number of successfull tasks migrated.
     * This is very important as this value is deducted from the total tasks 
     * the dealer device will execute
     * 
     */
    int deal_success = 0;

    find_compute_tasks(&(dealer_device->pending), dealer_device, starving_device,
                       TASK_MIGRATED_BEFORE_STAGE_IN, -1, DATA_REUSE_SELECTION, execution_level, ring, &tries, &deal_success);

    if (tries < parsec_cuda_migrate_chunk_size)
    {
        // level1 - task is availble in the stage_in queue. Stage_in not started.
        execution_level = 1;
        find_compute_tasks(dealer_device->exec_stream[0]->fifo_pending, dealer_device, starving_device,
                           TASK_MIGRATED_BEFORE_STAGE_IN, -1, DATA_REUSE_SELECTION, execution_level, ring, &tries, &deal_success);

        for (j = 0; j < (dealer_device->num_exec_streams - 2); j++)
        {
            if (tries < parsec_cuda_migrate_chunk_size)
            {
                // level2 - task is available in one of the execution queue stage_in is complete
                execution_level = 2;
                find_compute_tasks(dealer_device->exec_stream[(2 + j)]->fifo_pending, dealer_device, starving_device,
                                   TASK_MIGRATED_AFTER_STAGE_IN, -1, DATA_REUSE_SELECTION, execution_level, ring, &tries, &deal_success);
            }
            else
                break;
        }
    }

    (void)es;
    return deal_success;
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
    if (is_starving(dealer_device_index))
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
            migrated_gpu_task = (parsec_gpu_task_t *)parsec_list_pop_front(ring);
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
            for (k = 0; k < MAX_PARAM_COUNT; k++) migrated_gpu_task->candidate[i] = NULL;
            mig_task->dealer_device = dealer_device;
            mig_task->starving_device = starving_device;
            mig_task->stage_in_status = migrated_gpu_task->migrate_status;
        #if defined(PARSEC_PROF_TRACE)
            migrated_gpu_task->select_time = time_stamp();
        #endif

            PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t *)mig_task);
            parsec_cuda_mig_task_enqueue(es, mig_task);

            char tmp[MAX_TASK_STRLEN];
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream, "Task %s migrated (level %d, stage_in %d) from device %d to device %d",
                                 parsec_task_snprintf(tmp, MAX_TASK_STRLEN, ((parsec_gpu_task_t *)migrated_gpu_task)->ec),
                                 execution_level, mig_task->stage_in_status, dealer_device_index, starving_device_index);
        } // end while

        if (deal_success > 0)
        {
            device_info[dealer_device_index].success_count++;
            device_info[dealer_device_index].last_device = starving_device_index;
        }
        else
        {
            break;
        }

        if (is_starving(dealer_device_index))
        {
            break;
        }
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
            if ( !(PARSEC_FLOW_ACCESS_READ & gpu_task->flow[i]->flow_flags) && !(PARSEC_FLOW_ACCESS_READ & gpu_task->flow[i]->flow_flags) ) // CTL flow
                continue;

            parsec_data_t *original = task->data[i].data_out->original;
            parsec_atomic_lock(&original->lock);

            /** Read only access to data_out*/
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

                /**
                 * As the data has atleast one reader, one of which is this task being
                 * migrated, this data will not be evicted. So this is atleast in the
                 * list gpu_mem_owned_lru. If some other task is writing to it (which
                 * should not happen) this will not be in any tracking list, but the tasks
                 * that is writing to it will eventually add it to the appropriate tracking 
                 * list during parsec_cuda_kernel_epilog.
                 */

               
                /** The data in used in the first stage-in may have been evicted.
                 * But we store the original data_in and that can be used for the stage_in.
                 */
                assert(gpu_task->original_data_in[i] != NULL);
                if( gpu_task->original_data_in[i] != NULL && task->data[i].data_in == NULL)
                {
                    task->data[i].data_in = gpu_task->original_data_in[i];
                }
        
            }
           
            /** Write only access to data_out */
            else if (!(PARSEC_FLOW_ACCESS_READ & gpu_task->flow[i]->flow_flags) &&
                (PARSEC_FLOW_ACCESS_WRITE & gpu_task->flow[i]->flow_flags))
            {
                assert(task->data[i].data_out->readers == 0);
                assert(task->data[i].data_out->super.super.obj_reference_count == 1);
                
                /**
                 * @brief When the data is write access, is not in any tracking lists.
                 * As we won't need this data (as the data is write only and only
                 * this task is supposed to be write to this data) and it doesnt have any other readers
                 * we can add it to the gpu_mem_lru for eviction. 
                 * 
                 * But we decided that as D2D trasfers are more fast, compared to H2D transfers. 
                 * So we can use this data as the candidate for the next stage-in. 
                 * As this data has no reader this may endup being evicted during zone_malloc().
                 * So we increase the reader for this data. This reader will get decremented
                 * after the second stage-in.
                 */
                gpu_task->candidate[i] = task->data[i].data_out;
                PARSEC_DATA_COPY_INC_READERS_ATOMIC(task->data[i].data_out);

                parsec_list_item_ring_chop((parsec_list_item_t *)task->data[i].data_out);
                PARSEC_LIST_ITEM_SINGLETON(task->data[i].data_out);
                parsec_list_push_back(&dealer_device->gpu_mem_lru, (parsec_list_item_t *)task->data[i].data_out);

                /** The data in used in the first stage-in may have been released.
                 * But we store the original data_in and that can be used for the stage_in.
                 * As the flow is write-only we dont care about the version of the data.
                 */
                assert(gpu_task->original_data_in[i] != NULL);
                if( gpu_task->original_data_in[i] != NULL && task->data[i].data_in == NULL)
                {
                    task->data[i].data_in = gpu_task->original_data_in[i];
                }
            }
            
            /** Read and write access to data_out*/
            else if ((PARSEC_FLOW_ACCESS_READ & gpu_task->flow[i]->flow_flags) &&
                (PARSEC_FLOW_ACCESS_WRITE & gpu_task->flow[i]->flow_flags))
            {
                assert(task->data[i].data_out->readers == 1);
                /**
                 * we set a possible candidate for this flow of the task. This will allow
                 * us to easily find the stage_in data as the possible candidate in
                 * parsec_gpu_data_stage_in() function.
                 */
                gpu_task->candidate[i] = task->data[i].data_out;

                /**
                 * @brief As the task has write access, it is not in any tracking lists.
                 * As the data has atleast one reader, one of which is this task being
                 * migrated, this data will not be evicted. So we add it to gpu_mem_lru.
                 * As this task has write access, no ther task has write access to it.
                 * So when we add it to gpu_mem_lru, we are not in danger of prematurely 
                 * evicting the data, to whichn some other task has read or write access.
                 */
                parsec_list_item_ring_chop((parsec_list_item_t *)task->data[i].data_out);
                PARSEC_LIST_ITEM_SINGLETON(task->data[i].data_out);
                parsec_list_push_back(&dealer_device->gpu_mem_lru, (parsec_list_item_t *)task->data[i].data_out); 

                /** The data in used in the first stage-in may have been released.
                 * But we store the original data_in and that can be used for the stage_in.
                 */
                assert(gpu_task->original_data_in[i] != NULL);
                if( gpu_task->original_data_in[i] != NULL && task->data[i].data_in == NULL)
                {
                    task->data[i].data_in = gpu_task->original_data_in[i];
                }
            }
            else
            {
                assert(0);
            }

            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                "Migrate: data %p attached to original %p [readers %d, ref_count %d] migrated from device %d to %d (stage_in: %d)",
                task->data[i].data_out, original, task->data[i].data_out->readers,
                task->data[i].data_out->super.super.obj_reference_count, dealer_device->super.device_index,
                starving_device->super.device_index, TASK_MIGRATED_AFTER_STAGE_IN);

            parsec_atomic_unlock(&original->lock);
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

int inc_manager_complete_count(int device_index)
{
    parsec_atomic_fetch_inc_int32(&device_info[device_index].completed_manager);
    return device_info[device_index].completed_manager;
}

int inc_co_manager_complete_count(int device_index)
{
    parsec_atomic_fetch_inc_int32(&device_info[device_index].completed_co_manager);
    return device_info[device_index].completed_co_manager;
}

int inc_iterative_mapped_count(int device_index)
{
    parsec_atomic_fetch_inc_int32(&device_info[device_index].iterative_mapped);
    return device_info[device_index].iterative_mapped;
}

int inc_thrashing_count(int device_index)
{
    parsec_atomic_fetch_inc_int32(&device_info[device_index].thrashing);
    return device_info[device_index].thrashing;
}


int find_task_affinity_to_starving_node(parsec_gpu_task_t *gpu_task, int device_index, int status)
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

int find_task_to_task_affinity(parsec_gpu_task_t *first_gpu_task, parsec_gpu_task_t *sec_gpu_task, int status)
{
    int i, j, affinity = 0;
    parsec_task_t *first_task = NULL;
    parsec_task_t *sec_task = NULL;
    parsec_data_copy_t *first_task_data_copy = NULL;
    parsec_data_copy_t *sec_task_data_copy = NULL;

    /**
     * @brief first_gpu_task is NULL implies there is no task selected for migration.
     * So make sec_gpu_task task the first to be selected task; 
     */
    if( first_gpu_task == NULL )
        return 1;
    
    first_task = first_gpu_task->ec;
    sec_task = sec_gpu_task->ec;

    for (i = 0; i < sec_task->task_class->nb_flows; i++)
    {
        if (NULL == sec_task->data[i].data_in || NULL == sec_task->data[i].source_repo_entry)
            continue;

        if (status == TASK_MIGRATED_BEFORE_STAGE_IN) // data will be trasferred from data_in
            sec_task_data_copy = sec_task->data[i].data_in;
        else
            sec_task_data_copy = sec_task->data[i].data_out;

        for (j = 0; j < first_task->task_class->nb_flows; j++)
        {
            if (NULL == first_task->data[j].data_in || NULL == sec_task->data[i].source_repo_entry)
                continue;

            if (first_gpu_task->migrate_status == TASK_MIGRATED_BEFORE_STAGE_IN) // data will be trasferred from data_in
                first_task_data_copy = first_task->data[j].data_in;
            else
                first_task_data_copy = first_task->data[j].data_out;

            if( sec_task_data_copy == first_task_data_copy)
            {
                affinity = 1;
                break;  
            }
        }

        return affinity;
    }

    return affinity;
}

parsec_hook_return_t
parsec_cuda_co_manager( parsec_execution_stream_t *es, parsec_device_gpu_module_t* gpu_device )
{
    int rc = 0, nb_migrated = 0, i = 0;
    parsec_task_t* task = NULL;
    parsec_gpu_task_t *gpu_task = NULL;

    parsec_list_t *gpu_tasks_to_free = NULL; 

    (void)es;

    if( gpu_device->co_manager_mutex > 0 ) 
        return PARSEC_HOOK_RETURN_ASYNC;
    else 
    {
        rc = gpu_device->co_manager_mutex;
        if( !parsec_atomic_cas_int32( &gpu_device->co_manager_mutex, rc, rc+1 ) ) 
            return PARSEC_HOOK_RETURN_ASYNC;
    }

    gpu_tasks_to_free = PARSEC_OBJ_NEW(parsec_list_t);

    /**
     * @brief The migrate_manager thread exits when there are no more
     * work to be done.
     */
    while( gpu_device->mutex > 0 || gpu_device->complete_mutex > 0)
    {
       if(parsec_cuda_migrate_tasks == 2)
        {
            nb_migrated = migrate_to_starving_device(es,  gpu_device);
            if( nb_migrated > 0 )   
            {
                rc = parsec_atomic_fetch_add_int32(&(gpu_device->mutex), -1 * nb_migrated);
            }
        }

        if(gpu_device->complete_mutex > 0)
        {

            /** try completion PARSEC_MAX_EVENTS_PER_STREAM tasks */ 
            for( i = 0; i < PARSEC_MAX_EVENTS_PER_STREAM; i++)
            {
                gpu_task = NULL;
                task = NULL;

                gpu_task = (parsec_gpu_task_t*)parsec_fifo_pop( &(gpu_device->to_complete) );
                if( gpu_task != NULL)
                {
                    task = gpu_task->ec;
                
                #if defined(PARSEC_PROF_TRACE)
                     gpu_task->complete_time_start = time_stamp();
                #endif

                    __parsec_complete_execution( es, task );
                
                #if defined(PARSEC_PROF_TRACE)
                     gpu_task->complete_time_end = time_stamp();
                #endif

                #if defined(PARSEC_PROF_TRACE)
                    gpu_dev_prof_t prof_info;
                    prof_info.task_type                  = gpu_task->task_type;
                    prof_info.device_index               = gpu_device->super.device_index;
                    prof_info.task_count                 = gpu_device->mutex;
                    prof_info.first_queue_time           = gpu_task->first_queue_time;
                    prof_info.select_time                = gpu_task->select_time;
                    prof_info.second_queue_time          = gpu_task->second_queue_time;
                    prof_info.exec_time_start            = gpu_task->exec_time_start;
                    prof_info.exec_time_end              = gpu_task->exec_time_end;
                    prof_info.complete_time_start        = gpu_task->complete_time_start;
                    prof_info.complete_time_end          = gpu_task->complete_time_end;
                    prof_info.first_stage_in_time_start  = gpu_task->first_stage_in_time_start;
                    prof_info.sec_stage_in_time_start    = gpu_task->sec_stage_in_time_start;
                    prof_info.first_stage_in_time_end    = gpu_task->first_stage_in_time_end;
                    prof_info.sec_stage_in_time_end      = gpu_task->sec_stage_in_time_end;
                    prof_info.stage_out_time_start       = gpu_task->stage_out_time_start;
                    prof_info.stage_out_time_end         = gpu_task->stage_out_time_end;
                    prof_info.first_waiting_tasks        = gpu_task->first_waiting_tasks;
                    prof_info.sec_waiting_tasks          = gpu_task->sec_waiting_tasks;
                    prof_info.mig_status                 = gpu_task->migrate_status;
                    prof_info.nb_first_stage_in          = gpu_task->nb_first_stage_in;
                    prof_info.nb_sec_stage_in            = gpu_task->nb_sec_stage_in;
                    prof_info.nb_first_stage_in_d2d      = gpu_task->nb_first_stage_in_d2d;
                    prof_info.nb_first_stage_in_h2d      = gpu_task->nb_first_stage_in_h2d;
                    prof_info.nb_sec_stage_in_d2d        = gpu_task->nb_sec_stage_in_d2d;
                    prof_info.nb_sec_stage_in_h2d        = gpu_task->nb_sec_stage_in_h2d;
                    prof_info.clock_speed                = gpu_task->clock_speed;
                    prof_info.class_id                   = gpu_task->ec->task_class->task_class_id;
                    prof_info.exec_stream_index          = gpu_task->exec_stream_index;
                    parsec_profiling_trace_flags(es->es_profile,
                        parsec_gpu_task_count_end,
                        (uint64_t)gpu_task->ec->task_class->key_functions->key_hash(gpu_task->ec->task_class->make_key(gpu_task->ec->taskpool, gpu_task->ec->locals), NULL),
                        gpu_task->ec->taskpool->taskpool_id, &prof_info, 0);
                #endif 

                    parsec_atomic_fetch_dec_int32( &(gpu_device->complete_mutex) );

                    int f = 0;
                    for( f = 0; f < gpu_task->ec->task_class->nb_flows; f++)
                    {
                        if( gpu_task->original_data_in[f] != NULL )
                        {
                            PARSEC_OBJ_RELEASE( gpu_task->original_data_in[f] );
                        }
                    }

                    parsec_list_push_back(gpu_tasks_to_free, (parsec_list_item_t*)gpu_task);
                }                                  

                if( gpu_device->complete_mutex == 0 )
                {
                    break;
                }
            }
        }
        
    }
    
    rc = parsec_atomic_fetch_dec_int32( &(gpu_device->co_manager_mutex) );

    /** We free the task delegated to the co-manager only at the end. Or else it may
     * interfere with some operations in the manager.
    */
    while(NULL != (gpu_task = (parsec_gpu_task_t*)parsec_list_pop_front(gpu_tasks_to_free)) ) 
    {
        free(gpu_task);
    }

    PARSEC_OBJ_RELEASE(gpu_tasks_to_free);

    return PARSEC_HOOK_RETURN_ASYNC;
}