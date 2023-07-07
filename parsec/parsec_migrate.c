#include <time.h>
#include <stdlib.h>

#include "parsec/parsec_migrate.h"
#include "parsec/mca/device/cuda/device_cuda_migrate.h"

extern parsec_mempool_t *parsec_remote_dep_cb_data_mempool;
extern parsec_execution_stream_t parsec_comm_es;
extern parsec_sched_module_t *parsec_current_scheduler;
extern int parsec_communication_engine_up;
extern int parsec_runtime_node_migrate_tasks;
extern int parsec_runtime_node_migrate_stats;
extern int parsec_runtime_steal_request_policy;
extern int parsec_runtime_starvation_policy;
extern int parsec_runtime_chunk_size;
extern int parsec_device_cuda_enabled;
extern int parsec_runtime_node_migrate_stats;
extern int parsec_runtime_skew_distribution;
extern int parsec_runtime_starving_devices;
extern int parsec_runtime_hop_count;
extern int parsec_runtime_progress_count;
extern int parsec_runtime_task_mapping;

int finalised_hop_count  = 0;  /* Max hop count of a steal request */

parsec_list_t mig_noobj_fifo;               /* fifo of mig task details with taskpools not actually known */
parsec_list_t steal_req_fifo;               /** list of all steal request received */
parsec_list_t received_task_fifo;           /** list of all tasks received */
parsec_list_t direct_msg_fifo;               /* fifo of direct msg with taskpools not actually known */


/**
 * parsec_migration_engine_up == 0 : there is no node level task migration
 * parsec_migration_engine_up == 1 : node level task migration is active
 **/
int parsec_migration_engine_up = -1;

/**
 * @brief Ony one  steal request is send per node. This variable decides
 * of there are already any active steal request from this node in the system.
 **/
volatile int32_t active_steal_request_mutex = 0;

/**
 * @brief Only one thread processes a steal request at any given time. This decides
 * if the steal request processing is on going.
 **/
volatile int32_t process_steal_request_mutex = 0;

/** Keep track of the last victim to which the steal req was send*/
volatile int last_victim = -1;

/** counts the number of tasks progressed in a device after it
 * migrated the task.
*/
int* device_progress_counter = NULL;

parsec_node_info_t *node_info; /** stats on migration in a node */
static int my_rank, nb_nodes;
PARSEC_OBJ_CLASS_INSTANCE(migrated_node_level_task_t, migrated_node_level_task_t, NULL, NULL);

/** hashtable for storing task mapping */
static parsec_hash_table_t *task_map_ht = NULL; 
/** hashtable for storing details of received tasks */
static parsec_hash_table_t *received_task_ht = NULL; 
/** hashtable for storing details of migrated tasks */
static parsec_hash_table_t *migrated_task_ht = NULL; 
/** hashtable of direct messages */
static parsec_hash_table_t *direct_msg_ht = NULL; 

static parsec_key_fn_t node_task_mapping_table_generic_key_fn = {
    .key_equal = parsec_hash_table_generic_64bits_key_equal,
    .key_hash = parsec_hash_table_generic_64bits_key_hash,
    .key_print = parsec_hash_table_generic_64bits_key_print};

static void task_mapping_free_elt(void *_item, void *table);
void *migrate_engine_main(parsec_taskpool_t *tp);
static int recieve_mig_task_details(parsec_comm_engine_t *ce, parsec_ce_tag_t tag,
                                    void *msg, size_t msg_size, int src,
                                    void *cb_data);
static int remote_dep_get_datatypes_of_mig_task(parsec_execution_stream_t *es,
                                                parsec_remote_deps_t *origin);
static int get_mig_task_data_cb(parsec_comm_engine_t *ce,
                                parsec_ce_tag_t tag, void *msg, size_t msg_size,
                                int src, void *cb_data);
static int recieve_steal_request(parsec_comm_engine_t *ce, parsec_ce_tag_t tag,
                                 void *msg, size_t msg_size, int src,
                                 void *cb_data);
static void get_mig_task_data(parsec_execution_stream_t *es,
                              parsec_remote_deps_t *deps);
static parsec_remote_deps_t *get_mig_task_data_complete(parsec_execution_stream_t *es,
                                                        int idx, parsec_remote_deps_t *origin);
parsec_remote_deps_t *prepare_remote_deps(parsec_execution_stream_t *es,
                                          parsec_task_t *mig_task, int dst_rank, int src_rank);
static int migrate_dep_mpi_save_put_cb(parsec_comm_engine_t *ce, parsec_ce_tag_t tag, void *msg, size_t msg_size,
                                       int src, void *cb_data);
static void migrate_dep_mpi_put_start(parsec_execution_stream_t *es, dep_cmd_item_t *item);
static int migrate_dep_mpi_put_end_cb(parsec_comm_engine_t *ce, parsec_ce_mem_reg_handle_t lreg, ptrdiff_t ldispl,
                                      parsec_ce_mem_reg_handle_t rreg, ptrdiff_t rdispl, size_t size,
                                      int remote, void *cb_data);
parsec_remote_deps_t*  prepare_task_details_msg(parsec_execution_stream_t *es, parsec_task_t *this_task, int root,
                             void* buf, int length, int *position);
int  migrated_task_cleanup(parsec_execution_stream_t *es, parsec_gpu_task_t *gpu_task);
int progress_steal_request(parsec_execution_stream_t *es, steal_request_t *steal_request);
int get_gpu_wt_tasks(parsec_device_gpu_module_t * device);
static int update_task_mapping_cb(parsec_comm_engine_t *ce, parsec_ce_tag_t tag,
                         void *msg, size_t msg_size, int src, void *cb_data);
static int mig_dep_send(parsec_execution_stream_t* es, int rank, parsec_remote_deps_t *deps);
static int remote_direct_activate_cb(parsec_comm_engine_t *ce, parsec_ce_tag_t tag,
    void *msg, size_t msg_size, int src, void *cb_data);
static parsec_remote_deps_t*
mig_direct_release_incoming(parsec_execution_stream_t* es, parsec_remote_deps_t* origin,    
remote_dep_datakey_t complete_mask);
static void mig_direct_get_end(parsec_execution_stream_t* es, int idx, parsec_remote_deps_t* deps);
static int mig_direct_get_end_cb(parsec_comm_engine_t *ce, parsec_ce_tag_t tag, void *msg,
    size_t msg_size, int src, void *cb_data);
int mig_dep_direct_send(parsec_execution_stream_t* es, int rank, parsec_remote_deps_t *deps);
static void mig_direct_get_start(parsec_execution_stream_t* es, parsec_remote_deps_t* deps);
static void mig_direct_recv_activate(parsec_execution_stream_t* es, parsec_remote_deps_t* deps, 
    char* packed_buffer,int length, int* position);
static int mig_direct_activate_cb(parsec_comm_engine_t *ce, parsec_ce_tag_t tag,
    void *msg, size_t msg_size, int src, void *cb_data);
static int mig_direct_get_datatypes(parsec_execution_stream_t* es, parsec_remote_deps_t* origin,
    int storage_id, int *position);
parsec_ontask_iterate_t
mig_dep_mpi_retrieve_datatype(parsec_execution_stream_t *eu,
    const parsec_task_t *newcontext, const parsec_task_t *oldcontext,
    const parsec_dep_t* dep, parsec_dep_data_description_t* out_data,
    int src_rank, int dst_rank, int dst_vpid, data_repo_t *successor_repo, 
    parsec_key_t successor_repo_key, void *param);
static int mig_no_put_cb(parsec_comm_engine_t *ce, parsec_ce_tag_t tag,
    void *msg, size_t msg_size, int src, void *cb_data);
static void mig_direct_no_get_start(parsec_execution_stream_t* es,
                                     parsec_remote_deps_t* deps);
static int 
check_deps_received(parsec_execution_stream_t* es, parsec_remote_deps_t* origin);

PARSEC_DECLSPEC PARSEC_OBJ_CLASS_DECLARATION(steal_request_t);
PARSEC_OBJ_CLASS_INSTANCE(steal_request_t, parsec_list_item_t, NULL, NULL);

int get_nb_nodes() 
{
    return nb_nodes;
}

int parsec_node_mig_inc_gpu_task_executed()
{
    parsec_atomic_fetch_inc_int32(&(node_info->nb_gpu_tasks_executed));
    return node_info->nb_gpu_tasks_executed;
}

int parsec_node_mig_inc_cpu_task_executed()
{
    parsec_atomic_fetch_inc_int32(&(node_info->nb_cpu_tasks_executed));
    return node_info->nb_cpu_tasks_executed;
}


int parsec_node_mig_inc_released()
{
    parsec_atomic_fetch_inc_int32(&(node_info->nb_release));
    return node_info->nb_release;
}

int parsec_node_mig_inc_selected()
{
    parsec_atomic_fetch_inc_int32(&(node_info->nb_selected ));
    return node_info->nb_selected ;
} 

int parsec_node_mig_inc_req_send()
{
    parsec_atomic_fetch_inc_int32(&(node_info->nb_req_send));
    return node_info->nb_req_send;
}

int parsec_node_mig_inc_req_recvd()
{
    parsec_atomic_fetch_inc_int32(&(node_info->nb_req_recvd));
    return node_info->nb_req_recvd;
}

int parsec_node_mig_inc_req_forwarded()
{
    parsec_atomic_fetch_inc_int32(&(node_info->nb_req_forwarded));
    return node_info->nb_req_forwarded;
}

int parsec_node_mig_inc_req_processed()
{
    parsec_atomic_fetch_inc_int32(&(node_info->nb_req_processed));
    return node_info->nb_req_processed;
}

int parsec_node_mig_inc_success_req_processed()
{
    parsec_atomic_fetch_inc_int32(&(node_info->nb_succesfull_req_processed));
    return node_info->nb_succesfull_req_processed;
}

int parsec_node_mig_inc_task_migrated()
{
    parsec_atomic_fetch_inc_int32(&(node_info->nb_task_migrated));
    return node_info->nb_task_migrated;
}

int parsec_node_mig_inc_task_recvd()
{
    parsec_atomic_fetch_inc_int32(&(node_info->nb_task_recvd));
    return node_info->nb_task_recvd;
}

int parsec_node_mig_inc_success_steals(steal_request_t *steal_request)
{
    int hops_completed = 0;

    parsec_atomic_fetch_inc_int32(&(node_info->nb_succesfull_steals));

    if ((3 == parsec_runtime_steal_request_policy) /* RING HOPS */ || (4 == parsec_runtime_steal_request_policy) /* RANDOM HOPS */
            || (2 == parsec_runtime_steal_request_policy) /* LastVictim */) {

        hops_completed = finalised_hop_count - steal_request->msg.hop_count;
        parsec_atomic_fetch_add_int32(&(node_info->hops_succesfull_steals), hops_completed);
    }

    return node_info->nb_succesfull_steals;

}

int parsec_node_mig_inc_success_full_steals(steal_request_t *steal_request)
{
    int hops_completed = 0;

    parsec_atomic_fetch_inc_int32(&(node_info->nb_succesfull_full_steals));

    if ((3 == parsec_runtime_steal_request_policy) /* RING HOPS */ || (4 == parsec_runtime_steal_request_policy) /* RANDOM HOPS */
            || (2 == parsec_runtime_steal_request_policy) /* LastVictim */)  {

        hops_completed = finalised_hop_count - steal_request->msg.hop_count;
        parsec_atomic_fetch_add_int32(&(node_info->hops_succesfull_full_steals), hops_completed);
    }

    return node_info->nb_succesfull_full_steals;

}

int parsec_node_mig_inc_searches()
{
    parsec_atomic_fetch_inc_int32(&(node_info->nb_searches));
    return node_info->nb_searches;
}

int parsec_node_mig_inc_full_yield()
{
    parsec_atomic_fetch_inc_int32(&(node_info->full_yield));
    return node_info->full_yield;
}

int get_gpu_wt_tasks(parsec_device_gpu_module_t * device)
{
    return device->wt_tasks;
}

int nb_launched_task()
{
    int d = 0;
    int total_tasks = 0 ;
    parsec_device_gpu_module_t *gpu_device = NULL;

    for (d = 0; d < parsec_device_cuda_enabled; d++) {
        gpu_device = (parsec_device_gpu_module_t *)parsec_mca_device_get(DEVICE_NUM(d));
        total_tasks += gpu_device->mutex;
    }

    return total_tasks;
}



int parsec_node_migrate_init(parsec_context_t *context)
{
    int rc;

    if ((parsec_runtime_chunk_size * SINGLE_ACTIVATE_MSG_SIZE)  > MAX_ACTIVATE_MSG_SIZE) {
        printf("Supports a maximum chunk size of %d task. You can change this in parsec/parsec_migrate.h \n", MAX_CHUNK_SIZE);
        exit(0);
    }

    my_rank = context->my_rank;
    nb_nodes = context->nb_nodes;

    srand(time(NULL));

    rc = parsec_ce.tag_register(PARSEC_MIG_TASK_DETAILS_TAG, recieve_mig_task_details, context, MAX_ACTIVATE_MSG_SIZE * sizeof(char));
    if (PARSEC_SUCCESS != rc) {
        parsec_warning("[CE] Failed to register communication tag PARSEC_MIG_TASK_DETAILS_TAG (error %d)\n", rc);
        parsec_ce.tag_unregister(PARSEC_MIG_TASK_DETAILS_TAG);
        parsec_comm_engine_fini(&parsec_ce);
        return rc;
    }
    rc = parsec_ce.tag_register(PARSEC_MIG_STEAL_REQUEST_TAG, recieve_steal_request, context,
                                STEAL_REQ_MSG_SIZE * sizeof(char));
    if (PARSEC_SUCCESS != rc) {
        parsec_warning("[CE] Failed to register communication tag PARSEC_MIG_STEAL_REQUEST_TAG (error %d)\n", rc);
        parsec_ce.tag_unregister(PARSEC_MIG_STEAL_REQUEST_TAG);
        parsec_comm_engine_fini(&parsec_ce);
        return rc;
    }
    rc = parsec_ce.tag_register(PARSEC_MIG_DEP_GET_DATA_TAG, migrate_dep_mpi_save_put_cb, context,
                                4096);
    if (PARSEC_SUCCESS != rc) {
        parsec_warning("[CE] Failed to register communication tag PARSEC_MIG_DEP_GET_DATA_TAG (error %d)\n", rc);
        parsec_ce.tag_unregister(PARSEC_MIG_DEP_GET_DATA_TAG);
        parsec_comm_engine_fini(&parsec_ce);
        return rc;
    }
    rc = parsec_ce.tag_register(PARSEC_MIG_INFORM_PREDECESSOR_TAG, update_task_mapping_cb, context, MAPPING_INFO_SIZE * sizeof(char));
    if (PARSEC_SUCCESS != rc) {
        parsec_warning("[CE] Failed to register communication tag PARSEC_MIG_INFORM_PREDECESSOR_TAG (error %d)\n", rc);
        parsec_ce.tag_unregister(PARSEC_MIG_INFORM_PREDECESSOR_TAG);
        parsec_comm_engine_fini(&parsec_ce);
        return rc;
    }
    rc = parsec_ce.tag_register(PARSEC_MIG_DEP_DIRECT_ACTIVATE_TAG, mig_direct_activate_cb, context, SINGLE_ACTIVATE_MSG_SIZE * sizeof(char));
    if (PARSEC_SUCCESS != rc) {
        parsec_warning("[CE] Failed to register communication tag PARSEC_MIG_DEP_DIRECT_ACTIVATE_TAG (error %d)\n", rc);
        parsec_ce.tag_unregister(PARSEC_MIG_DEP_DIRECT_ACTIVATE_TAG);
        parsec_comm_engine_fini(&parsec_ce);
        return rc;
    }
    rc = parsec_ce.tag_register(PARSEC_MIG_NO_GET_DATA_TAG, mig_no_put_cb, context,
                                SINGLE_ACTIVATE_MSG_SIZE * sizeof(char) );
    if (PARSEC_SUCCESS != rc) {
        parsec_warning("[CE] Failed to register communication tag PARSEC_MIG_NO_GET_DATA_TAG (error %d)\n", rc);
        parsec_ce.tag_unregister(PARSEC_MIG_NO_GET_DATA_TAG);
        parsec_comm_engine_fini(&parsec_ce);
        return rc;
    }

    finalised_hop_count = ((0 == parsec_runtime_hop_count) || (parsec_runtime_hop_count >= nb_nodes) ) ? (nb_nodes - 1) : parsec_runtime_hop_count;

    if (parsec_communication_engine_up > 0) {
        parsec_migration_engine_up = 1;
    }

    device_progress_counter = (int *)calloc(parsec_device_cuda_enabled, sizeof(int));

    PARSEC_OBJ_CONSTRUCT(&steal_req_fifo, parsec_list_t);
    PARSEC_OBJ_CONSTRUCT(&mig_noobj_fifo, parsec_list_t);
    PARSEC_OBJ_CONSTRUCT(&received_task_fifo, parsec_list_t);
    PARSEC_OBJ_CONSTRUCT(&direct_msg_fifo, parsec_list_t);
    

    if(parsec_runtime_task_mapping) {
        task_map_ht = PARSEC_OBJ_NEW(parsec_hash_table_t);
        parsec_hash_table_init(task_map_ht, offsetof(mig_task_mapping_item_t, ht_item), 16, 
            node_task_mapping_table_generic_key_fn, NULL);

        received_task_ht = PARSEC_OBJ_NEW(parsec_hash_table_t);
        parsec_hash_table_init(received_task_ht, offsetof(mig_task_mapping_item_t, ht_item), 16, 
            node_task_mapping_table_generic_key_fn, NULL);

        migrated_task_ht = PARSEC_OBJ_NEW(parsec_hash_table_t);
        parsec_hash_table_init(migrated_task_ht, offsetof(mig_task_mapping_item_t, ht_item), 16, 
            node_task_mapping_table_generic_key_fn, NULL);

        direct_msg_ht = PARSEC_OBJ_NEW(parsec_hash_table_t);
        parsec_hash_table_init(direct_msg_ht, offsetof(mig_task_mapping_item_t, ht_item), 16, 
            node_task_mapping_table_generic_key_fn, NULL);     
    }
    
    return 0;
}

int parsec_node_task_count_start;
int parsec_node_task_count_end;
int parsec_all_task_count_start;
int parsec_all_task_count_end;
int parsec_steal_req_recv_start;
int parsec_steal_req_recv_end;
int parsec_steal_req_send_start;
int parsec_steal_req_send_end;
int parsec_steal_req_init_start;
int parsec_steal_req_init_end;

static void node_profiling_init()
{
    parsec_profiling_add_dictionary_keyword("NODE_GPU_TASK_COUNT", "fill:#FF0000", sizeof(node_prof_t), "ready_tasks{double};complete_time{double}",
                                            &parsec_node_task_count_start, &parsec_node_task_count_end);
    
    parsec_profiling_add_dictionary_keyword("NODE_ALL_TASK_COUNT", "fill:#FF0000", sizeof(node_prof_t), "tp_nb_tasks{double};task_progress{double}",
                                            &parsec_all_task_count_start, &parsec_all_task_count_end);

    parsec_profiling_add_dictionary_keyword("REQ_RECVD", "fill:#FF0000", sizeof(steal_req_prof_t), "gpu_tasks{double};req_recv_time{double}",
                                            &parsec_steal_req_recv_start, &parsec_steal_req_recv_end);

    parsec_profiling_add_dictionary_keyword("REQ_SEND", "fill:#FF0000", sizeof(steal_req_prof_t), "launched_tasks{double};req_send_time{double}",
                                            &parsec_steal_req_send_start, &parsec_steal_req_send_end);

    parsec_profiling_add_dictionary_keyword("REQ_INIT", "fill:#FF0000", sizeof(steal_req_prof_t), "req_mutex{double};req_init_time{double}",
                                            &parsec_steal_req_init_start, &parsec_steal_req_init_end);
}

int parsec_node_stats_init(parsec_context_t *context)
{
    my_rank = context->my_rank;
    nb_nodes = context->nb_nodes;
    
    node_info = (parsec_node_info_t *)calloc(1, sizeof(parsec_node_info_t));
    node_info->nb_gpu_tasks_executed            = 0;
    node_info->nb_task_recvd                    = 0;
    node_info->nb_task_migrated                 = 0;
    node_info->nb_req_recvd                     = 0;
    node_info->nb_req_send                      = 0; 
    node_info->nb_req_send                      = 0;
    node_info->nb_req_processed                 = 0;
    node_info->nb_succesfull_req_processed      = 0;
    node_info->nb_searches                      = 0;
    node_info->nb_req_forwarded                 = 0;
    node_info->full_yield                       = 0;
    node_info->nb_release                       = 0;
    node_info->nb_selected                      = 0;
    node_info->nb_succesfull_steals             = 0;
    node_info->nb_succesfull_full_steals        = 0;

#if defined(PARSEC_PROF_TRACE)
    node_profiling_init();
#endif
    
    return 0;
}

int parsec_node_stats_fini()
{
    printf("\n*********** NODES %d/%d *********** \n", my_rank + 1, nb_nodes);
    printf("GPU Tasks executed              : %d \n", node_info->nb_gpu_tasks_executed);
    printf("CPU Tasks executed              : %d \n", node_info->nb_cpu_tasks_executed);
    printf("Tasks released                  : %d \n", node_info->nb_release );
    printf("\n");

    printf("Steal req received              : %d \n", node_info->nb_req_recvd);
    printf("Steal req processed             : %d \n", node_info->nb_req_processed);
    printf("Successful req processed        : %d \n", node_info->nb_succesfull_req_processed);
    printf("Total full yield                : %d \n", node_info->full_yield);
    printf("Perc successful req processed   : %lf \n", ((double)node_info->nb_succesfull_req_processed / (double)node_info->nb_req_recvd) * 100);
    printf("Perc successful full yield      : %lf \n", ((double)node_info->full_yield / (double)node_info->nb_req_recvd) * 100);
    printf("Tasks migrated                  : %d \n", node_info->nb_task_migrated);
    printf("\n");

    printf("Steal req send                  : %d \n", node_info->nb_req_send);
    printf("Successfull steals              : %d \n", node_info->nb_succesfull_steals);
    printf("Successfull full steals         : %d \n", node_info->nb_succesfull_full_steals);
    printf("Perc successful steals          : %lf \n", ((double)node_info->nb_succesfull_steals / (double)node_info->nb_req_send) * 100);
    printf("Perc successful full steals     : %lf \n", ((double)node_info->nb_succesfull_full_steals / (double)node_info->nb_req_send) * 100);
    printf("Task recvd                      : %d \n", node_info->nb_task_recvd);
    printf("\n");
    
    printf("Chunk size                      : %d \n", parsec_runtime_chunk_size); 
    printf("Starving policy (#device)       : %d \n", parsec_runtime_starving_devices); 
    printf("Starvation policy (#tasks)      : %d \n", parsec_runtime_starvation_policy);
    printf("Progress count (#tasks)         : %d \n", parsec_runtime_progress_count);
    printf("\n");
    
    if (0 == parsec_runtime_steal_request_policy) {
        printf("Steal req policy                : Ring \n");
    }
    else if (1 == parsec_runtime_steal_request_policy) {
        printf("Steal req policy                : Random \n");
    }
    else if (2 == parsec_runtime_steal_request_policy) {
        printf("Steal req policy                : LastVictim \n");
    }
    else if (3 == parsec_runtime_steal_request_policy) {
        printf("Steal req policy                : RingHops \n");
    }
    else if (4 == parsec_runtime_steal_request_policy) {
        printf("Steal req policy                : RandomHops \n");
    }

    printf("Finalised hopcount              : %d \n", finalised_hop_count);
    printf("Avg. hop per success. steal     : %lf \n", ((double)node_info->hops_succesfull_steals / (double)node_info->nb_succesfull_steals) );
    printf("Avg. hop per full success. steal: %lf \n", ((double)node_info->hops_succesfull_full_steals / (double)node_info->nb_succesfull_full_steals) );
    free(node_info);

    if (0 == parsec_runtime_skew_distribution) {
        printf("Data distrbution                : Normal \n");
    }
    else {
        printf("Data distrbution                : Skewed \n");
    }
    printf("Total Nodes                     : %d \n", nb_nodes);

    return 0;
}

int parsec_node_migrate_fini()
{
    parsec_list_item_t *item;

    parsec_migration_engine_up = 0;

    while ((item = parsec_list_pop_front(&steal_req_fifo)) != NULL) { free(item); }
    PARSEC_OBJ_DESTRUCT(&steal_req_fifo);
    PARSEC_OBJ_DESTRUCT(&mig_noobj_fifo);
    PARSEC_OBJ_DESTRUCT(&received_task_fifo);
    PARSEC_OBJ_DESTRUCT(&direct_msg_fifo);

    parsec_ce.tag_unregister(PARSEC_MIG_TASK_DETAILS_TAG);
    parsec_ce.tag_unregister(PARSEC_MIG_STEAL_REQUEST_TAG);
    parsec_ce.tag_unregister(PARSEC_MIG_DEP_GET_DATA_TAG);
    parsec_ce.tag_unregister(PARSEC_MIG_INFORM_PREDECESSOR_TAG);
    parsec_ce.tag_unregister(PARSEC_MIG_DEP_DIRECT_ACTIVATE_TAG);
    parsec_ce.tag_unregister(PARSEC_MIG_NO_GET_DATA_TAG);
    
    free(device_progress_counter);

    if(parsec_runtime_task_mapping) {
        parsec_hash_table_for_all(task_map_ht, task_mapping_free_elt, task_map_ht);
        parsec_hash_table_fini(task_map_ht);
        PARSEC_OBJ_RELEASE(task_map_ht);
        task_map_ht = NULL;

        parsec_hash_table_for_all(received_task_ht, task_mapping_free_elt, received_task_ht);
        parsec_hash_table_fini(received_task_ht);
        PARSEC_OBJ_RELEASE(received_task_ht);
        received_task_ht = NULL;

        parsec_hash_table_for_all(migrated_task_ht, task_mapping_free_elt, migrated_task_ht);
        parsec_hash_table_fini(migrated_task_ht);
        PARSEC_OBJ_RELEASE(migrated_task_ht);
        migrated_task_ht = NULL;

        parsec_hash_table_for_all(direct_msg_ht, task_mapping_free_elt, direct_msg_ht);
        parsec_hash_table_fini(direct_msg_ht);
        PARSEC_OBJ_RELEASE(direct_msg_ht);
        direct_msg_ht = NULL;

        
    }

    return parsec_migration_engine_up;
}

int increment_progress_counter(int device_num)
{
    device_progress_counter[device_num]++;
}

int unset_progress_counter(int device_num)
{
    device_progress_counter[device_num] = 0;
}

int get_progress_counter(int device_num)
{
    return device_progress_counter[device_num];
}

int schedule_migrated_task(parsec_execution_stream_t* es, parsec_task_t *task)
{
    parsec_list_push_back(&received_task_fifo, (parsec_list_item_t *)task);
    return 1;
}

parsec_task_t* select_migrated_task(parsec_execution_stream_t* es)
{
    parsec_task_t *task = NULL;
    task = (parsec_task_t *)parsec_list_pop_front(&received_task_fifo);
    return task;
}

int progress_migrated_task(parsec_execution_stream_t* es)
{
    parsec_task_t *task = NULL;
    task = (parsec_task_t *)select_migrated_task(&received_task_fifo);

    if(NULL != task) {
        __parsec_task_progress(es, task, 1);
        return 1;
    }

    return 0; 
}

static int
recieve_steal_request(parsec_comm_engine_t *ce, parsec_ce_tag_t tag,
                      void *msg, size_t msg_size, int src,
                      void *cb_data)
{
    (void)tag;
    (void)cb_data;
    (void)ce;
    (void)msg_size;
    (void)src;

    steal_request_t *steal_request = NULL;
    steal_request_msg_t* req_msg = NULL;

    int array_pos = 0;
    int array_mask = 0;
    int current_mask = 0;
    int k = 0;
    int max_index = 0;
   
    req_msg = (steal_request_msg_t *)msg;

    assert( STEAL_REQ_MSG_SIZE == msg_size );

    steal_request = PARSEC_OBJ_NEW(steal_request_t);
    memcpy(&(steal_request->msg), req_msg, STEAL_REQ_MSG_SIZE);

    assert(0 <= steal_request->msg.root && steal_request->msg.root < nb_nodes);
    assert(0 <= steal_request->msg.src  && steal_request->msg.src < nb_nodes);
    assert(0 <= steal_request->msg.dst  && steal_request->msg.dst < nb_nodes);

    if (steal_request->msg.root == my_rank) /** request initiated from this node */ {
        if(steal_request->msg.nb_task_request < parsec_runtime_chunk_size) {
            if(0 == steal_request->msg.nb_task_request) {
                parsec_node_mig_inc_success_full_steals(steal_request);
            }
            parsec_node_mig_inc_success_steals(steal_request);

        }

        if (2 == parsec_runtime_steal_request_policy) {
            assert(last_victim != -1);

            array_pos = last_victim / MAX_NODES_INDEX;
            current_mask = steal_request->msg.failed_victims[array_pos];
            array_mask  = 0;
            array_mask |= 1 << (last_victim % RANKS_PER_INDEX);

            if( (current_mask & array_mask) > 0) {
                last_victim = -1;
            }

            /** Find if there is a successfull node in the neighbourhood
             * of the last victim. Ideally we should search for the successfull
             * node. But if the total nodes are large, this may be a costly operation.
            */
            max_index = (array_pos == (nb_nodes / MAX_NODES_INDEX) )   /** last victim is the same index as the nb_nodes*/
                        ? (nb_nodes % RANKS_PER_INDEX) /** Index is limited to the index of the nb_nodes  */
                        : MAX_NODES_INDEX; /** use the max index*/
            array_pos = last_victim / MAX_NODES_INDEX;

            for(k = 0;  k < max_index; k++) {
                array_mask = 0;
                array_mask |= (1 << k);
                current_mask = steal_request->msg.successful_victims[array_pos];

                if( (current_mask & array_mask) > 0) {
                    last_victim = (array_pos * RANKS_PER_INDEX) + k;
                    break;
                }
            }
        }

        parsec_atomic_fetch_dec_int32(&active_steal_request_mutex);
        PARSEC_OBJ_RELEASE(steal_request);
    }
    else {
        parsec_list_push_back(&steal_req_fifo, (parsec_list_item_t *)steal_request);

        if (parsec_runtime_node_migrate_stats) {
            parsec_node_mig_inc_req_recvd();
        }

        PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "MIG-DEBUG: Steal request %p recvd from rank %d on rank %d. #task requested %d",
                             steal_request, steal_request->msg.src, steal_request->msg.dst, steal_request->msg.nb_task_request);
    }

    return 0;
}


int process_steal_request(parsec_execution_stream_t *es)
{
    int d = 0, rc = 0, total_selected = 0, device_selected = 0;
    int tasks_requested = 0;
    steal_request_t *steal_request = NULL;
    parsec_device_gpu_module_t *gpu_device = NULL;
    parsec_gpu_task_t *gpu_task = NULL;
    parsec_task_t *task;
    parsec_list_t *list = NULL;
    parsec_list_item_t *item = NULL;
    int nb_cuda_devices = parsec_device_cuda_enabled;
    migrated_node_level_task_t *mig_task = NULL;
    int array_pos = 0;
    int array_mask = 0;
    void *buff = NULL;
    
    if (0 != process_steal_request_mutex /* someone already processing*/
        || parsec_list_nolock_is_empty(&steal_req_fifo) /* no steal request*/) {

        return PARSEC_HOOK_RETURN_ASYNC;
    }
    
    if (!parsec_atomic_cas_int32(&process_steal_request_mutex, 0, 1)) {  
        return PARSEC_HOOK_RETURN_ASYNC;
    }
    
    do {
        steal_request = (steal_request_t *)parsec_list_pop_front(&steal_req_fifo);

        if(NULL == steal_request) {
            break;
        }
    
        assert(0 <= steal_request->msg.root && steal_request->msg.root < nb_nodes);
        assert(0 <= steal_request->msg.src  && steal_request->msg.src < nb_nodes);
        assert(0 <= steal_request->msg.dst  && steal_request->msg.dst < nb_nodes);

        tasks_requested = steal_request->msg.nb_task_request;

    #if defined(PARSEC_PROF_TRACE)
        steal_req_prof_t steal_prof;
        double ready_tasks = 0;

        parsec_profiling_trace_flags(es->es_profile,parsec_steal_req_recv_start,
            (uint64_t)steal_request, parsec_device_cuda_enabled, NULL, 0);


        for (d = 0; d < parsec_device_cuda_enabled; d++) {
            gpu_device = (parsec_device_gpu_module_t *)parsec_mca_device_get(DEVICE_NUM(d));            
            ready_tasks += gpu_device->mutex; 
        }

        steal_prof.gpu_tasks = ready_tasks;
        steal_prof.recv_time = time_stamp();

        parsec_profiling_trace_flags(es->es_profile, parsec_steal_req_recv_end,
            (uint64_t)steal_request, parsec_device_cuda_enabled, &steal_prof, 0);
    #endif

        if (parsec_runtime_node_migrate_stats) {
            parsec_node_mig_inc_req_processed();
        }
        
        parsec_list_t *ring = PARSEC_OBJ_NEW(parsec_list_t);
        PARSEC_OBJ_RETAIN(ring);

        total_selected = 0; /** reset for each steal requests */
        for (d = 0; d < nb_cuda_devices; d++) {
            if( 0 != active_steal_request_mutex /* I am a thief */) {
                break;
            }
                    
            device_selected = 0; /** reset for each device */
            gpu_device = (parsec_device_gpu_module_t *)parsec_mca_device_get(DEVICE_NUM(d));

            if ( (gpu_device->wt_tasks > parsec_runtime_starvation_policy) && (get_progress_counter(d) > parsec_runtime_progress_count ) ) {
                list = &(gpu_device->pending);
                if( !parsec_atomic_trylock( &list->atomic_lock ) ) {
                    continue;
                }

                for (item = PARSEC_LIST_ITERATOR_FIRST(list);
                     (PARSEC_LIST_ITERATOR_END(list) != item);
                     item = PARSEC_LIST_ITERATOR_NEXT(item)) {

                    gpu_task = (parsec_gpu_task_t *)item;

                    if ((gpu_task != NULL) && (gpu_task->task_type == PARSEC_GPU_TASK_TYPE_KERNEL) &&
                        (gpu_task->ec->mig_status != PARSEC_MIGRATED_TASK) &&
                        (gpu_task->ec->task_class->task_class_id == 5)) {

                        item = parsec_list_nolock_remove(list, item);
                        PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t *)gpu_task);
                        parsec_list_nolock_push_back(ring, (parsec_list_item_t *)gpu_task);

                        parsec_node_mig_inc_selected();
                        total_selected++;
                        device_selected++;
                    }

                    if (total_selected >= tasks_requested) {
                        break;
                    }
                }
                
                rc = parsec_atomic_fetch_add_int32(&(gpu_device->mutex), (-1 * device_selected));
                parsec_atomic_fetch_add_int32( &(gpu_device->wt_tasks), (-1 * device_selected));

                if(device_selected > 0) {
                    unset_progress_counter(d);
                }
                if (parsec_runtime_node_migrate_stats) {
                    parsec_node_mig_inc_searches();
                }

                parsec_atomic_unlock( &list->atomic_lock );

                
            }

            if (total_selected >= tasks_requested) {
                if (parsec_runtime_node_migrate_stats) {
                    parsec_node_mig_inc_full_yield();
                }
                break;
            }
        }


        if( !parsec_list_nolock_is_empty(ring) ) {
            parsec_node_mig_inc_success_req_processed();

            int position = 0 /** starting position of the message*/, buffer_size = 0;

            parsec_list_t *tasks_to_free = PARSEC_OBJ_NEW(parsec_list_t);
            PARSEC_OBJ_RETAIN(tasks_to_free);
        
            buffer_size = total_selected * SINGLE_ACTIVATE_MSG_SIZE * sizeof(char); 
            /** buff holds the message that will be send to the thief node */
            buff = malloc(buffer_size);

            for (item = PARSEC_LIST_ITERATOR_FIRST(ring);
                (PARSEC_LIST_ITERATOR_END(ring) != item);
                item = PARSEC_LIST_ITERATOR_NEXT(item)) {

                gpu_task = (parsec_gpu_task_t *)item;
                item = parsec_list_nolock_remove(ring, item);
                PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t *)gpu_task);

                if (NULL != gpu_task) {
                    /** for each task add the details of the task to be migrated to the buff */
                    parsec_remote_deps_t *deps = NULL;
                    deps = prepare_task_details_msg(es, gpu_task->ec, steal_request->msg.root, buff, buffer_size, &position );

                    if(parsec_runtime_task_mapping) {
                        /** send the new task mapping to the predecessors*/
                        send_task_mapping_info_to_predecessor(es, gpu_task->ec, steal_request->msg.root);
                        /** insert the details of the migrated task to a HT */
                        insert_migrated_tasks_details(gpu_task->ec, steal_request->msg.root);
                    }

                    /** release the resources held by the migrated task */
                    migrated_task_cleanup(es, gpu_task);
                    free(gpu_task); 
                }
            }
        
            /** send the whole buffer to the thief node. It contains details about 'total_selected' number of tasks */
            parsec_ce.send_am(&parsec_ce, PARSEC_MIG_TASK_DETAILS_TAG, steal_request->msg.root, buff, position);
            free(buff);
        }

        PARSEC_OBJ_RELEASE(ring);
        ring = NULL;
        
       
        if (2 == parsec_runtime_steal_request_policy) /* LAST VICTIM */ {
            array_pos = my_rank / MAX_NODES_INDEX;
            array_mask = 1 << (my_rank % RANKS_PER_INDEX);

            if( total_selected == tasks_requested) {
                /** mark this node a successfull s*/
                steal_request->msg.successful_victims[array_pos] |= array_mask;
            }
            else if( 0 == total_selected) {
                /** mark this node as failure */
                steal_request->msg.failed_victims[array_pos] |= array_mask;
            }
        }

        steal_request->msg.nb_task_request -= total_selected;

        progress_steal_request(es, steal_request);
        PARSEC_OBJ_RELEASE(steal_request);
          
    } while( (1 == parsec_migration_engine_up) );

    process_steal_request_mutex = 0;
    return 1;
}


parsec_remote_deps_t* prepare_task_details_msg(parsec_execution_stream_t *es, parsec_task_t *this_task, int root,
    void* packed_buffer, int length, int *position)
{
    parsec_remote_deps_t *deps = NULL;
    int src_rank = 0, dst_rank = 0;
    int dsize = 0, saved_position = *position;

    PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "MIG-DEBUG: Task %p selected for migration", this_task);

    /** deps will be send to the node that initiated the request */
    dst_rank = root;
    src_rank = my_rank;

    deps = prepare_remote_deps(es, this_task, dst_rank, src_rank);
    assert(deps->taskpool != NULL && this_task->taskpool == deps->taskpool);

    /** Size of the data to be send  */
    parsec_ce.pack_size(&parsec_ce, SINGLE_ACTIVATE_MSG_SIZE, parsec_datatype_int8_t, &dsize);

    /** one task details  size. It is important to set this here, so as to be used on the receiver */
    deps->msg.length = dsize; 

    if( (length - saved_position) < dsize ) {
        printf("There is something wrong with the buffer allocation length = %d, position %d, dsize = %d \n", 
        length, *position, dsize);
        assert(0);
    }

    /** pack the message to the buffer starting at position*/
    parsec_ce.pack(&parsec_ce, &deps->msg, SINGLE_ACTIVATE_MSG_SIZE, parsec_datatype_int8_t, packed_buffer, length, &saved_position);
    /** Update the position. Next task details will start from this position*/
    *position = *position + dsize;

    /** This will be decremented by remote_dep_complete_and_cleanup() which is called
     * in migrate_dep_mpi_put_end_cb after() each PUT.
    */
    remote_dep_inc_flying_messages(deps->taskpool);

    /** TODO: This shouldnt be here.
     * Ideally there should be only one increment per message. But incrementing one
     * per deps makes the logic easier on the receiver side. 
    */
    deps->taskpool->tdm.module->outgoing_message_start(deps->taskpool, dst_rank, deps);


    PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "MIG-DEBUG: Migration reply send to rank %d using deps %p with pending ack %d",
                         dst_rank, deps, deps->pending_ack);

    return deps;
}

int migrated_task_cleanup(parsec_execution_stream_t *es, parsec_gpu_task_t *gpu_task)
{
    parsec_task_t *this_task = gpu_task->ec;
    int i = 0;
    /**
     * @brief Release everything owned by this task, in the similar manner if it was executed on this node
     *  this_task->task_class->release_task() will decrease the task count on this node.
     */

    for (i = 0; i < this_task->task_class->nb_flows; i++) {
        if (this_task->task_class->in[i] == NULL) {
            continue;
        }

        /** Make sure the flow is either READ/WRITE or READ and not a CTL flow*/
        if ((this_task->task_class->in[i]->flow_flags & PARSEC_FLOW_ACCESS_MASK) == PARSEC_FLOW_ACCESS_NONE ) {
            continue;
        }

        /** If the repo associated with a data is not NULL reduce the usage count by one.*/
        if (this_task->data[i].source_repo_entry != NULL) {
            data_repo_entry_used_once(this_task->data[i].source_repo, this_task->data[i].source_repo_entry->ht_item.key);
        }
    }

    /** If the tasks 'consumes' a local repo reduce the usage count by one.*/
    if (this_task->repo_entry != NULL) {
        data_repo_entry_used_once(this_task->taskpool->repo_array[this_task->task_class->task_class_id], this_task->repo_entry->ht_item.key);
    }

    for (i = 0; i < this_task->task_class->nb_flows; i++) {
        if (this_task->task_class->in[i] == NULL) {
            continue;
        }

        /** Make sure the flow is either READ/WRITE or READ and not a CTL flow*/
        if (( this_task->task_class->in[i]->flow_flags & PARSEC_FLOW_ACCESS_MASK) == PARSEC_FLOW_ACCESS_NONE ) {
            continue;
        }

        if (NULL != this_task->data[i].data_in) {
            PARSEC_DATA_COPY_RELEASE(this_task->data[i].data_in);
        }
    }

    /**
     * 1. release everything related to this task in this node
     * 2. decrement the task count
     * */
    this_task->task_class->release_task(es, this_task);

    if (parsec_runtime_node_migrate_stats) {
        parsec_node_mig_inc_task_migrated();
    }

    return 0;
}

int initiate_steal_request(parsec_execution_stream_t *es)
{
    int i = 0;
    int victim_rank = 0;

    steal_request_msg_t steal_request_msg;

    steal_request_msg.nb_task_request = parsec_runtime_chunk_size;
    steal_request_msg.root            = my_rank;
    steal_request_msg.src             = my_rank;
    steal_request_msg.hop_count       = finalised_hop_count;
    for( i = 0; i < MAX_NODES_INDEX; i++) {
        steal_request_msg.failed_victims[i] = 0;
        steal_request_msg.successful_victims[i] = 0;
    }


    if (0 == parsec_runtime_steal_request_policy) { /* RING */
        steal_request_msg.dst = (my_rank + 1) % nb_nodes;
    }
    else if (1 == parsec_runtime_steal_request_policy) { /* RANDOM */
        victim_rank = rand() % nb_nodes;

        if (victim_rank == my_rank) {
            victim_rank = (victim_rank + 1) % nb_nodes;
        }

        steal_request_msg.dst = victim_rank;
    }
    else if (2 == parsec_runtime_steal_request_policy) /* LAST VICTIM */
    {
        if( -1 == last_victim ) {
            victim_rank = rand() % nb_nodes;

            if (victim_rank == my_rank) {
                victim_rank = (victim_rank + 1) % nb_nodes;
            }
            steal_request_msg.dst = victim_rank;

            /* Store the the current victim as the last victim*/
            last_victim = victim_rank; 
        }
        else {
            steal_request_msg.dst = last_victim;
        }
        steal_request_msg.hop_count -= 1;
    }
    else if (3 == parsec_runtime_steal_request_policy) { /* RING HOPS */
        steal_request_msg.dst = (my_rank + 1) % nb_nodes;
        steal_request_msg.hop_count -= 1;
    }
    else if (4 == parsec_runtime_steal_request_policy) { /* RANDOM HOPS */
        victim_rank = rand() % nb_nodes;

        if (victim_rank == my_rank) {
            victim_rank = (victim_rank + 1) % nb_nodes;
        }

        steal_request_msg.dst = victim_rank;
        steal_request_msg.hop_count -= 1;
    }
    else {
        printf("Wrong steal policy option \n");
        exit(0);
    }

#if defined(PARSEC_PROF_TRACE)
    steal_req_prof_t steal_prof;

    parsec_profiling_trace_flags(es->es_profile,
    parsec_steal_req_send_start,
    (uint64_t)(&steal_request_msg),
    parsec_device_cuda_enabled, NULL, 0);


    steal_prof.gpu_tasks = nb_launched_task();
    steal_prof.recv_time = time_stamp();

    parsec_profiling_trace_flags(es->es_profile,
        parsec_steal_req_send_end,
        (uint64_t)(&steal_request_msg),
        parsec_device_cuda_enabled, &steal_prof, 0);
#endif

    parsec_ce.send_am(&parsec_ce, PARSEC_MIG_STEAL_REQUEST_TAG, steal_request_msg.dst, &steal_request_msg, STEAL_REQ_MSG_SIZE);
    PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "MIG-DEBUG: Steal request message %p send to rank %d from rank %d. #task requested %d",
                         &steal_request_msg, steal_request_msg.dst, steal_request_msg.src, steal_request_msg.nb_task_request);

    if (parsec_runtime_node_migrate_stats) {
        parsec_node_mig_inc_req_send();
    }

    return 0;
}

int send_steal_request(parsec_execution_stream_t *es)
{
    int i, rc;
    steal_request_t steal_request;

#if defined(PARSEC_PROF_TRACE)
    steal_req_prof_t steal_prof;

    parsec_profiling_trace_flags(es->es_profile,
    parsec_steal_req_init_start,
    (uint64_t)(&steal_request),
    parsec_device_cuda_enabled, NULL, 0);


    steal_prof.gpu_tasks = active_steal_request_mutex;
    steal_prof.recv_time = time_stamp();

    parsec_profiling_trace_flags(es->es_profile,
        parsec_steal_req_init_end,
        (uint64_t)(&steal_request),
        parsec_device_cuda_enabled, &steal_prof, 0);
#endif

    if (parsec_migration_engine_up == 0 
        || active_steal_request_mutex != 0 /** steal request already sent */
        || !parsec_list_nolock_is_empty(&received_task_fifo) /** there are migrated task to process */) {

        return PARSEC_HOOK_RETURN_ASYNC;
    }

    rc = active_steal_request_mutex;

    if (!parsec_atomic_cas_int32(&active_steal_request_mutex, 0, 1)) {
        return PARSEC_HOOK_RETURN_ASYNC;
    }

    assert(1 == active_steal_request_mutex);
    
    initiate_steal_request(es);

    return PARSEC_HOOK_RETURN_ASYNC;
}

int nb_starving_device(parsec_execution_stream_t *es)
{
    int d = 0;
    int starving = 0 ;
    parsec_device_gpu_module_t *gpu_device = NULL;

    for (d = 0; d < parsec_device_cuda_enabled; d++) {
        gpu_device = (parsec_device_gpu_module_t *)parsec_mca_device_get(DEVICE_NUM(d));
        if( gpu_device->mutex < 1 ) {
            starving++;
        }
    }

    return starving;
}

int progress_steal_request(parsec_execution_stream_t *es, steal_request_t *steal_request)
{
    int victim_rank = 0;
    int array_pos  = 0;
    int array_mask = 0;
    int current_mask = 0;
    int try = 0;

    steal_request->msg.src = my_rank;

    if (0 == steal_request->msg.nb_task_request) {
        steal_request->msg.dst = steal_request->msg.root;
    }
    else
    {
        if (0 == parsec_runtime_steal_request_policy) { /* RING */
            steal_request->msg.dst = (my_rank + 1) % nb_nodes;
        }
        else if (1 == parsec_runtime_steal_request_policy) { /* RANDOM */
            steal_request->msg.dst = steal_request->msg.root;
        }
        else if (2 == parsec_runtime_steal_request_policy) { /* LAST VICTIM*/ 
            steal_request->msg.hop_count -= 1;

            if (0 == steal_request->msg.hop_count) {
                steal_request->msg.dst = steal_request->msg.root;
            }
            else {
                victim_rank = rand() % nb_nodes;
                if (victim_rank == my_rank) {
                    victim_rank = (victim_rank + 1) % nb_nodes;
                }
            }
        }
        else if (3 == parsec_runtime_steal_request_policy) { /* RING HOPS */
            steal_request->msg.hop_count -= 1;

            if (0 == steal_request->msg.hop_count) {
                steal_request->msg.dst = steal_request->msg.root;
            }
            else {
                steal_request->msg.dst = (my_rank + 1) % nb_nodes;
            }
        }
        else if (4 == parsec_runtime_steal_request_policy) { /* RANDOM HOPS */
            steal_request->msg.hop_count -= 1;

            if (0 == steal_request->msg.hop_count) {
                steal_request->msg.dst = steal_request->msg.root;
            }
            else {
                victim_rank = rand() % nb_nodes;

                if (victim_rank == my_rank)
                {
                    victim_rank = (victim_rank + 1) % nb_nodes;
                }
                steal_request->msg.dst = victim_rank;
            }
        }
    }

    assert(0 <= steal_request->msg.root && steal_request->msg.root < nb_nodes);
    assert(0 <= steal_request->msg.src  && steal_request->msg.src < nb_nodes);
    assert(0 <= steal_request->msg.dst  && steal_request->msg.dst < nb_nodes);

    parsec_ce.send_am(&parsec_ce, PARSEC_MIG_STEAL_REQUEST_TAG, steal_request->msg.dst, &(steal_request->msg), STEAL_REQ_MSG_SIZE);

    PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "MIG-DEBUG: Steal request %p forwarded to rank %d from rank %d. #task requested %d",
                         steal_request, steal_request->msg.dst, steal_request->msg.src, steal_request->msg.nb_task_request);

    return PARSEC_HOOK_RETURN_ASYNC;
}

parsec_remote_deps_t *prepare_remote_deps(parsec_execution_stream_t *es,
                                          parsec_task_t *mig_task, int dst_rank, int src_rank)
{

    parsec_remote_deps_t *deps = NULL;
    struct remote_dep_output_param_s *output = NULL;
    int i = 0, _array_mask = 0, _array_pos = 0;

    deps = remote_deps_allocate(&parsec_remote_dep_context.freelist);
    assert(deps != NULL);
    assert(0 == deps->pending_ack);

    deps->root = src_rank;
    deps->msg.task_class_id = mig_task->task_class->task_class_id;
    deps->msg.taskpool_id = mig_task->taskpool->taskpool_id;
    deps->msg.deps = (uintptr_t)deps;
    deps->msg.root = deps->root;
    deps->taskpool = parsec_taskpool_lookup(deps->msg.taskpool_id);

    assert(deps->taskpool == mig_task->taskpool);
    assert(NULL != deps->taskpool);

    for (i = 0; i < mig_task->task_class->nb_locals; i++) {
        deps->msg.locals[i] = mig_task->locals[i];
    }
    _array_mask = 1 << (dst_rank % (8 * sizeof(uint32_t)));
    _array_pos = dst_rank / (8 * sizeof(uint32_t));

    for (i = 0; i < mig_task->task_class->nb_flows; i++) {
        /**
         * @brief This is important as we will be using iterate_predecessors() in
         * remote_dep_get_datatypes_of_mig_task() and iterate_predecessors needs
         * data_out to well defined.
         */
        mig_task->data[i].data_out = mig_task->data[i].data_in;
    }

    remote_dep_get_datatypes_of_mig_task(es, deps);

    for (i = 0; i < mig_task->task_class->nb_flows; i++) {
        if (mig_task->task_class->in[i] == NULL) {
            continue;
        }

        /** Make sure the flow is either READ/WRITE or READ and not a CTL flow*/
        if (( mig_task->task_class->in[i]->flow_flags & PARSEC_FLOW_ACCESS_MASK) == PARSEC_FLOW_ACCESS_NONE ) {
            continue; 
        }

        assert(NULL != parsec_data_copy_get_ptr(mig_task->data[i].data_in));

        output = &deps->output[i];
        output->data.data = mig_task->data[i].data_in;
        PARSEC_OBJ_RETAIN(mig_task->data[i].data_in);
        output->rank_bits[_array_pos] |= _array_mask;
        output->count_bits++; /** This is required in remote_dep_complete_and_cleanup()*/
        output->deps_mask |= (1 << i);
        deps->outgoing_mask |= (1 << i);

        parsec_atomic_fetch_inc_int32(&deps->pending_ack);
    }

    /** This is important when we try to get the stored data */
    deps->msg.output_mask = deps->outgoing_mask;
    assert(deps->outgoing_mask != 0);

    return deps;
}

static int
recieve_mig_task_details(parsec_comm_engine_t *ce, parsec_ce_tag_t tag,
                         void *msg, size_t msg_size, int src,
                         void *cb_data)
{
    (void)tag;
    (void)cb_data;
    parsec_execution_stream_t *es = &parsec_comm_es;

    int position = 0, length = msg_size, rc;
    parsec_remote_deps_t *deps = NULL;

    assert( msg_size <= MAX_ACTIVATE_MSG_SIZE ); 
    assert( (msg_size % SINGLE_ACTIVATE_MSG_SIZE) == 0); 

    /** The message can contain details of more than one migrateed task. */
    while (position < length) {
        deps = remote_deps_allocate(&parsec_remote_dep_context.freelist);
        assert( NULL != deps );

        parsec_ce.unpack(&parsec_ce, msg, length, &position, &deps->msg, SINGLE_ACTIVATE_MSG_SIZE, parsec_datatype_int8_t);
        deps->from = src;
        deps->root = deps->msg.root;
        assert(deps->root == src);
        deps->eager_msg = NULL;

        rc = remote_dep_get_datatypes_of_mig_task(es, deps);
        
        if(-1 == rc) {
            parsec_list_push_back(&mig_noobj_fifo, (parsec_list_item_t*)deps);
        }
        else {
            get_mig_task_data(es, deps);
        }
    }

    assert(position == length);

    return 1;
}

void mig_new_taskpool(parsec_execution_stream_t* es, dep_cmd_item_t *dep_cmd_item)
{
    parsec_taskpool_t* obj = dep_cmd_item->cmd.new_taskpool.tp;
    parsec_list_item_t *item;
    int rc;

    parsec_list_lock(&mig_noobj_fifo);

    for(item = PARSEC_LIST_ITERATOR_FIRST(&mig_noobj_fifo);
        item != PARSEC_LIST_ITERATOR_END(&mig_noobj_fifo);
        item = PARSEC_LIST_ITERATOR_NEXT(item) ) {

        parsec_remote_deps_t* deps = (parsec_remote_deps_t*)item;

        if( deps->msg.taskpool_id == obj->taskpool_id ) {
            deps->taskpool = NULL;
            rc = remote_dep_get_datatypes_of_mig_task(es, deps); 

            assert( -1 != rc );
            assert(deps->taskpool != NULL);
        
            item = parsec_list_nolock_remove(&mig_noobj_fifo, item);
            get_mig_task_data(es, deps);
        }
    }

    parsec_list_unlock(&mig_noobj_fifo);

    parsec_list_lock(&direct_msg_fifo);

    for(item = PARSEC_LIST_ITERATOR_FIRST(&direct_msg_fifo);
        item != PARSEC_LIST_ITERATOR_END(&direct_msg_fifo);
        item = PARSEC_LIST_ITERATOR_NEXT(item) ) {

        parsec_remote_deps_t* deps = (parsec_remote_deps_t*)item;

        if( deps->msg.taskpool_id == obj->taskpool_id ) {
            char* buffer = (char*)deps->taskpool;  /* get back the buffer from the "temporary" storage */
            deps->taskpool = NULL;
            int rc, position = 0;
            rc = mig_direct_get_datatypes(es, deps, 0, &position); 
            assert( -1 != rc );

            rc = check_deps_received(es, deps);
            if ( -2 == rc) {
                /** The same message was received from someone else */
                mig_direct_no_get_start(es, deps);
                free(deps);
            }
            else {
                item = parsec_list_nolock_remove(&direct_msg_fifo, item);

                mig_direct_recv_activate(es, deps, buffer,
                                     position + deps->msg.length, &position);
            }
            
        }
    }

    parsec_list_unlock(&direct_msg_fifo);

    
}

static int remote_dep_get_datatypes_of_mig_task(parsec_execution_stream_t *es,
                                                parsec_remote_deps_t *deps)
{
    int i = 0, flow_index = 0, rc = 0;
    parsec_task_t task;
    uint32_t flow_mask = 0;
    struct remote_dep_output_param_s *output = NULL;

    deps->taskpool = parsec_taskpool_lookup(deps->msg.taskpool_id);

    if(NULL == deps->taskpool) {
        return -1;
    }

    task.taskpool = deps->taskpool;
    task.task_class = task.taskpool->task_classes_array[deps->msg.task_class_id];
    for (i = 0; i < task.task_class->nb_locals; i++) {
        task.locals[i] = deps->msg.locals[i];
    }

    for (flow_index = 0; flow_index < task.task_class->nb_flows; flow_index++) {
        if (task.task_class->in[flow_index] == NULL) {
            continue;
        }

        /** Make sure the flow is either READ/WRITE or READ and not a CTL flow*/
        if ( (task.task_class->in[flow_index]->flow_flags & PARSEC_FLOW_ACCESS_MASK) == PARSEC_FLOW_ACCESS_NONE ) {
            continue;
        }

        flow_mask = (1U << task.task_class->in[flow_index]->flow_index) | 0x80000000U;
        output = &deps->output[flow_index];
        rc = task.task_class->get_datatype(es, &task, &flow_mask, &output->data);
        assert(PARSEC_HOOK_RETURN_NEXT == rc);
        output->data.data = NULL; /* This is important on the receiver side */

        assert(output->data.remote.src_datatype != PARSEC_DATATYPE_NULL);
        assert(output->data.remote.src_count != 0);
        assert(output->data.remote.arena != NULL);
    }

    return 1;
}

static inline parsec_data_copy_t *
migrated_copy_allocate(parsec_dep_type_description_t *data)
{
    parsec_data_copy_t *dc;
    if (NULL == data->arena) {
        assert(0 == data->dst_count);
        return NULL;
    }
    dc = parsec_arena_get_copy(data->arena, data->dst_count, 0, data->dst_datatype);

    dc->coherency_state = PARSEC_DATA_COHERENCY_EXCLUSIVE;
    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MIG-DEBUG: MPI:\tMalloc new remote tile %p size %" PRIu64 " count = %" PRIu64 " displ = %" PRIi64 " %p",
                         dc, data->arena->elem_size, data->dst_count, data->dst_displ, data->arena);
    return dc;
}

static void get_mig_task_data(parsec_execution_stream_t *es,
                              parsec_remote_deps_t *deps)
{
    remote_dep_wire_activate_t *task = &(deps->msg);
    int from = deps->from, k, nbdtt;
    remote_dep_wire_get_t msg;
    MPI_Datatype dtt;

    int rc = deps->taskpool->tdm.module->incoming_message_start(deps->taskpool, deps->from, &deps->msg, NULL /*not imp today*/,
                                                                0 /*not imp today*/, deps);

    assert(deps->msg.output_mask != 0);
    deps->incoming_mask = deps->msg.output_mask; /** This is important as we are changing deps->msg.output_mask soon */

    msg.source_deps = task->deps;                      /* the deps copied from activate message from source */
    assert(NULL != task->deps);
    msg.callback_fn = (uintptr_t)get_mig_task_data_cb; /* Function to call when PUT, in response to the GET is done */

    for (k = 0; deps->incoming_mask >> k; k++) {
        if (!((1U << k) & deps->incoming_mask)) {
            continue;
        }
        msg.output_mask = 0; /* Only get what I need */
        msg.output_mask |= (1U << k);

        /* We pack the callback data that should be passed to us when the other side
         * notifies us to invoke the callback_fn we have assigned above
         */
        remote_dep_cb_data_t *callback_data = (remote_dep_cb_data_t *)parsec_thread_mempool_allocate(parsec_remote_dep_cb_data_mempool->thread_mempools);
        callback_data->deps = deps;
        callback_data->k = k;

        deps->output[k].data.data = migrated_copy_allocate(&deps->output[k].data.remote);
        dtt = deps->output[k].data.remote.src_datatype;
        nbdtt = deps->output[k].data.remote.src_count;

        /* We have the remote mem_handle. Let's allocate our mem_reg_handle
         * and let the source know.
         */
        parsec_ce_mem_reg_handle_t receiver_memory_handle;
        size_t receiver_memory_handle_size;

        if (parsec_ce.capabilites.supports_noncontiguous_datatype) {
            parsec_ce.mem_register(PARSEC_DATA_COPY_GET_PTR(deps->output[k].data.data), PARSEC_MEM_TYPE_NONCONTIGUOUS,
                                   nbdtt, dtt,
                                   -1,
                                   &receiver_memory_handle, &receiver_memory_handle_size);
        }
        else {
            /* TODO: Implement converter to pack and unpack */
            int dtt_size;
            parsec_type_size(dtt, &dtt_size);
            parsec_ce.mem_register(PARSEC_DATA_COPY_GET_PTR(deps->output[k].data.data), PARSEC_MEM_TYPE_CONTIGUOUS,
                                   -1, NULL,
                                   dtt_size,
                                   &receiver_memory_handle, &receiver_memory_handle_size);
        }

        assert(NULL != receiver_memory_handle);
        assert(receiver_memory_handle_size == parsec_ce.get_mem_handle_size());

#if defined(PARSEC_DEBUG_NOISIER)
        char type_name[MPI_MAX_OBJECT_NAME];
        int len;
        char tmp[128];
        MPI_Type_get_name(dtt, type_name, &len);
        int _size;
        MPI_Type_size(dtt, &_size);
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "MIG-DEBUG: MPI:\tTO\t%d\tGet START\t% -8s\tk=%d\twith datakey %lx at %p type %s count %d displ %ld \t(k=%d, dst_mem_handle=%p)",
                             from, tmp, k, task->deps, PARSEC_DATA_COPY_GET_PTR(deps->output[k].data.data), type_name, dtt, nbdtt,
                             deps->output[k].data.remote.dst_displ, k, receiver_memory_handle);
#endif

        callback_data->memory_handle = receiver_memory_handle;

        /* We need multiple information to be passed to the callback_fn we have assigned above.
         * We pack the pointer to this callback_data and pass to the other side so we can complete
         * cleanup and take necessary action when the data is available on our side */
        msg.remote_callback_data = (remote_dep_datakey_t)callback_data;

        /* We pack the static message(remote_dep_wire_get_t) and our memory_handle and send this message
         * to the source. Source is anticipating this exact configuration.
         */
        int buf_size = sizeof(remote_dep_wire_get_t) + receiver_memory_handle_size;
        assert(buf_size == sizeof(remote_dep_wire_get_t) + parsec_ce.get_mem_handle_size());

        void *buf = malloc(buf_size);
        /** Copy the message to the buffer */
        memcpy(buf, &msg, sizeof(remote_dep_wire_get_t));
        /** Copy the handle to the rest of the buffer */
        memcpy(((char *)buf) + sizeof(remote_dep_wire_get_t), receiver_memory_handle, receiver_memory_handle_size);

        /* Send AM */
        parsec_ce.send_am(&parsec_ce, PARSEC_MIG_DEP_GET_DATA_TAG, from, buf, buf_size);
        free(buf);

        parsec_comm_gets++;
    }
}

static int
get_mig_task_data_cb(parsec_comm_engine_t *ce,
                     parsec_ce_tag_t tag,
                     void *msg,
                     size_t msg_size,
                     int src,
                     void *cb_data)
{

    (void)ce;
    (void)tag;
    (void)msg_size;
    (void)cb_data;
    (void)src;
    parsec_execution_stream_t *es = &parsec_comm_es;

    /* We send 8 bytes to the source to give it back to us when the PUT is completed,
     * let's retrieve that
     */
    uintptr_t *retrieve_pointer_to_callback = (uintptr_t *)msg;
    remote_dep_cb_data_t *callback_data = (remote_dep_cb_data_t *)*retrieve_pointer_to_callback;
    parsec_remote_deps_t *deps = (parsec_remote_deps_t *)callback_data->deps;

#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif

    PARSEC_DEBUG_VERBOSE(6, parsec_debug_output, "MIG-DEBUG: MPI:\tFROM\t%d\tGet END  \t% -8s\tk=%d\twith datakey na        \tparams %lx\t(tag=%d)",
                         src, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN),
                         callback_data->k, deps->incoming_mask, src);

    TAKE_TIME(es->es_profile, MPI_Data_pldr_ek, callback_data->k);
    get_mig_task_data_complete(es, callback_data->k, deps);

    parsec_ce.mem_unregister(&callback_data->memory_handle);
    parsec_thread_mempool_free(parsec_remote_dep_cb_data_mempool->thread_mempools, callback_data);
    parsec_comm_gets--;

    return 1;
}

static parsec_remote_deps_t *
get_mig_task_data_complete(parsec_execution_stream_t *es,
                           int idx,
                           parsec_remote_deps_t *origin)
{
    int i = 0, pidx = 0, flow_index = 0, distance = 0;
    remote_dep_datakey_t complete_mask = (1U << idx);
    parsec_task_t *task = NULL;
    parsec_key_t key;

    assert((origin->incoming_mask & complete_mask) == complete_mask);
    origin->incoming_mask ^= complete_mask;

    /**
     * @brief
     * when origin->incoming_mask == 0, it implies that all the data items of the task
     * has been recived and it will be available in origin->output[i].data.data indexed
     * by the flow index.
     */
    if (0 != origin->incoming_mask) /* not done receiving */ {
        return origin;
    }

    task = (parsec_task_t *)parsec_thread_mempool_allocate(es->context_mempool);
    task->taskpool = origin->taskpool;
    task->task_class = task->taskpool->task_classes_array[origin->msg.task_class_id];
    task->priority = origin->priority;
    for (i = 0; i < task->task_class->nb_locals; i++) {
        task->locals[i] = origin->msg.locals[i];
    }
    task->repo_entry = NULL;
    task->mig_status = PARSEC_MIGRATED_TASK;
    task->status = PARSEC_TASK_STATUS_EVAL; /** Skip the prepare input step */

    for (flow_index = 0; flow_index < task->task_class->nb_flows; flow_index++) {
        task->data[flow_index].source_repo = NULL;
        task->data[flow_index].source_repo_entry = NULL;
        task->data[flow_index].data_in = NULL;
        task->data[flow_index].data_out = NULL;

        if (task->task_class->in[flow_index] == NULL) {
            continue;
        }

        /** Make sure the flow is either READ/WRITE or READ and not a CTL flow*/
        if ( (task->task_class->in[flow_index]->flow_flags & PARSEC_FLOW_ACCESS_MASK) == PARSEC_FLOW_ACCESS_NONE ) {
            continue;
        }

        task->data[flow_index].data_in = origin->output[flow_index].data.data;
        task->data[flow_index].data_out = origin->output[flow_index].data.data;
        origin->output[flow_index].data.data->readers = 0;
    }

    /** mark the end of communication for this migration message */
    origin->taskpool->tdm.module->incoming_message_end(origin->taskpool, origin);

    /** Update the task count on this node */
    origin->taskpool->tdm.module->taskpool_addto_nb_tasks(origin->taskpool, 1);
    /** Schedule the task on this node */
    parsec_list_item_singleton((parsec_list_item_t *)task);
    PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "MIG-DEBUG: Received task %p scheduled for execution", task);
    task->chore_mask = PARSEC_DEV_ALL;

    if (parsec_runtime_node_migrate_stats) {
        parsec_node_mig_inc_task_recvd();
    }

    if(parsec_runtime_task_mapping) {
        insert_received_tasks_details(task, origin->root);
    }

    /** schedule the migrated tasks */
    schedule_migrated_task(es, task);
    assert(origin->root == origin->from);

    remote_deps_free(origin);
    return 0;
}

static int
migrate_dep_mpi_save_put_cb(parsec_comm_engine_t *ce, parsec_ce_tag_t tag, void *msg,
                            size_t msg_size, int src, void *cb_data)
{
    (void)ce;
    (void)tag;
    (void)cb_data;
    (void)msg_size;
    remote_dep_wire_get_t *task;
    parsec_remote_deps_t *deps;
    dep_cmd_item_t *item;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif
    parsec_execution_stream_t *es = &parsec_comm_es;

    /* we are expecting exactly one wire_get_t + remote memory handle */
    assert(msg_size == sizeof(remote_dep_wire_get_t) + ce->get_mem_handle_size());

    item = (dep_cmd_item_t *)malloc(sizeof(dep_cmd_item_t));
    PARSEC_OBJ_CONSTRUCT(&item->super, parsec_list_item_t);
    item->action = DEP_GET_DATA;
    item->cmd.activate.peer = src;

    task = &(item->cmd.activate.task);

    /* copy the static part of the message */
    memcpy(task, msg, sizeof(remote_dep_wire_get_t));

    /** the part after this contains the memory_handle of the other side.*/
    item->cmd.activate.remote_memory_handle = malloc(ce->get_mem_handle_size());
    memcpy(item->cmd.activate.remote_memory_handle, ((char *)msg) + sizeof(remote_dep_wire_get_t), 
        ce->get_mem_handle_size());

    deps = (parsec_remote_deps_t *)(remote_dep_datakey_t)task->source_deps; /* get our deps back */
    assert(0 != deps->pending_ack);
    assert(0 != deps->outgoing_mask);
    item->priority = deps->max_priority;

    PARSEC_DEBUG_VERBOSE(6, parsec_debug_output, "MIG-DEBUG: MPI: Put cb_received for %s from %d tag %u which 0x%x (deps %p)",
                         remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN), item->cmd.activate.peer,
                         -1, task->output_mask, (void *)deps);

   
    migrate_dep_mpi_put_start(es, item);
    
    return 1;
}



static void
migrate_dep_mpi_put_start(parsec_execution_stream_t *es, dep_cmd_item_t *item)
{
    remote_dep_wire_get_t *task = &(item->cmd.activate.task);
#if !defined(PARSEC_PROF_DRY_DEP)
    parsec_remote_deps_t *deps = (parsec_remote_deps_t *)(uintptr_t)task->source_deps;
    int k, nbdtt;
    void *dataptr;
    MPI_Datatype dtt;
#endif /* !defined(PARSEC_PROF_DRY_DEP) */
#if defined(PARSEC_DEBUG_NOISIER)
    char type_name[MPI_MAX_OBJECT_NAME];
    int len;
#endif

    (void)es;

#if !defined(PARSEC_PROF_DRY_DEP)
    assert(task->output_mask);
    PARSEC_DEBUG_VERBOSE(6, parsec_debug_output, "MIG-DEBUG: MPI:\tPUT mask=%lx deps 0x%lx", task->output_mask, task->source_deps);

    for (k = 0; task->output_mask >> k; k++) {
        assert(k < MAX_PARAM_COUNT);
        if (!((1U << k) & task->output_mask)) {
            continue;
        }

        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "MIG-DEBUG: MPI:\t[idx %d mask(0x%x / 0x%x)] %p, %p", k, (1U << k), task->output_mask,
                             deps->output[k].data.data, PARSEC_DATA_COPY_GET_PTR(deps->output[k].data.data));
        dataptr = PARSEC_DATA_COPY_GET_PTR(deps->output[k].data.data);
        dtt = deps->output[k].data.remote.src_datatype;
        nbdtt = deps->output[k].data.remote.src_count;
        (void)nbdtt;

        task->output_mask ^= (1U << k);

        parsec_ce_mem_reg_handle_t source_memory_handle;
        size_t source_memory_handle_size;

        if (parsec_ce.capabilites.supports_noncontiguous_datatype) {
            parsec_ce.mem_register(dataptr, PARSEC_MEM_TYPE_NONCONTIGUOUS,
                                   nbdtt, dtt,
                                   -1,
                                   &source_memory_handle, &source_memory_handle_size);
        }
        else {
            /* TODO: Implement converter to pack and unpack */
            int dtt_size;
            parsec_type_size(dtt, &dtt_size);
            parsec_ce.mem_register(dataptr, PARSEC_MEM_TYPE_CONTIGUOUS,
                                   -1, NULL, // TODO JS: this interface is so broken, fix it!
                                   dtt_size, // TODO JS: what about nbdtt? Is it ok to ignore it?!
                                   &source_memory_handle, &source_memory_handle_size);
        }

        parsec_ce_mem_reg_handle_t remote_memory_handle = item->cmd.activate.remote_memory_handle;
        assert( NULL != remote_memory_handle);
        assert( NULL != source_memory_handle);
        assert( source_memory_handle_size == parsec_ce.get_mem_handle_size() );

#if defined(PARSEC_DEBUG_NOISIER)
        MPI_Type_get_name(dtt, type_name, &len);
        PARSEC_DEBUG_VERBOSE(10, parsec_comm_output_stream, "MIG-DEBUG: MPI:\tTO\t%d\tPut START\tunknown \tk=%d\twith deps 0x%lx at %p type %s (%p)\t(src_mem_handle = %p, dst_mem_handle = %p)",
                             item->cmd.activate.peer, k, task->source_deps, dataptr, type_name, dtt, source_memory_handle, remote_memory_handle);
#endif

        remote_dep_cb_data_t *cb_data = (remote_dep_cb_data_t *)parsec_thread_mempool_allocate(parsec_remote_dep_cb_data_mempool->thread_mempools);
        cb_data->deps = deps;
        cb_data->k = k;

        TAKE_TIME_WITH_INFO(es->es_profile, MPI_Data_plds_sk, k,
                            es->virtual_process->parsec_context->my_rank,
                            item->cmd.activate.peer, deps->msg, nbdtt, dtt, MPI_COMM_WORLD);

        /* the remote side should send us 8 bytes as the callback data to be passed back to them */
        parsec_ce.put(&parsec_ce, source_memory_handle, 0,
                      remote_memory_handle, 0,
                      0, item->cmd.activate.peer,
                      migrate_dep_mpi_put_end_cb, cb_data,
                      (parsec_ce_tag_t)task->callback_fn, &task->remote_callback_data, sizeof(uintptr_t));

        parsec_comm_puts++;
    }
#endif /* !defined(PARSEC_PROF_DRY_DEP) */
    if (0 == task->output_mask) {
        if (NULL != item->cmd.activate.remote_memory_handle) {
            free(item->cmd.activate.remote_memory_handle);
            item->cmd.activate.remote_memory_handle = NULL;
        }
        free(item);
    }
}

static int migrate_dep_mpi_put_end_cb(parsec_comm_engine_t *ce, parsec_ce_mem_reg_handle_t lreg, ptrdiff_t ldispl,
                                      parsec_ce_mem_reg_handle_t rreg, ptrdiff_t rdispl, size_t size,
                                      int remote, void *cb_data)
{
    (void)ldispl;
    (void)rdispl;
    (void)size;
    (void)remote;
    (void)rreg;
    /* Retreive deps from callback_data */
    parsec_remote_deps_t *deps = ((remote_dep_cb_data_t *)cb_data)->deps;

    PARSEC_DEBUG_VERBOSE(6, parsec_debug_output, "MIG-DEBUG: MPI:\tTO\tna\tPut END  \tunknown \tk=%d\twith deps %p\tparams bla\t(src_mem_hanlde = %p, dst_mem_handle=%p",
                         ((remote_dep_cb_data_t *)cb_data)->k, deps, lreg, rreg);

#if defined(PARSEC_PROF_TRACE)
    TAKE_TIME(parsec_comm_es.es_profile, MPI_Data_plds_ek, ((remote_dep_cb_data_t *)cb_data)->k);
#endif

    remote_dep_complete_and_cleanup(&deps, 1);

    ce->mem_unregister(&lreg);
    parsec_thread_mempool_free(parsec_remote_dep_cb_data_mempool->thread_mempools, cb_data);

    parsec_comm_puts--;
    return 1;
}

parsec_dependency_t*
parsec_hash_find_sources(const parsec_taskpool_t *tp,
                      parsec_execution_stream_t *es,
                      const parsec_task_t* restrict task)
{
    parsec_hashable_sources_t *hs;
    parsec_key_handle_t kh;
    parsec_hash_table_t *ht = (parsec_hash_table_t*)tp->sources_array[task->task_class->task_class_id];

    parsec_key_t key = task->task_class->make_key(tp, task->locals);
    assert(NULL != ht);
    parsec_hash_table_lock_bucket_handle(ht, key, &kh);
    hs = parsec_hash_table_nolock_find_handle(ht, &kh);
    if( NULL == hs ) {
        hs = (parsec_hashable_sources_t *) malloc(sizeof(parsec_hashable_sources_t));
        hs->sources = (parsec_dependency_t)0;
        hs->ht_item.key = key;
        parsec_hash_table_nolock_insert_handle(ht, &kh, &hs->ht_item);
    }
    parsec_hash_table_unlock_bucket_handle(ht, &kh);
    return &(hs->sources);
}

parsec_dependency_t
parsec_update_sources(parsec_taskpool_t *tp, parsec_dependency_t *sources, int src)
{
    parsec_dependency_t source_new_value;

    assert(NULL != sources);

    if ( 0 != (*sources & (1 << src)) ) {
        /** this source has already been set */
        return *sources;
    }

    source_new_value = (*sources | (1 << src));
    parsec_atomic_fetch_or_int32( sources, source_new_value );
    return *sources;
}

static void task_mapping_free_elt(void *_item, void *table)
{
    mig_task_mapping_item_t *item = (mig_task_mapping_item_t *)_item;
    parsec_key_t key = item->ht_item.key;
    parsec_hash_table_nolock_remove(table, key);
    free(item);
}

int insert_migrated_tasks_details(parsec_task_t *task, int rank)
{
    parsec_key_t key;
    mig_task_mapping_item_t *item;

    assert(0 <= rank && rank < get_nb_nodes());
    key = task->task_class->make_key(task->taskpool, task->locals);

    /**
     * @brief Entry NULL imples that this task has never been received
     * till now in any of the iteration. So we start a new entry.
     */
    if (NULL == (item = parsec_hash_table_nolock_find(migrated_task_ht, key)))
    {
        item = (mig_task_mapping_item_t *)malloc(sizeof(mig_task_mapping_item_t));
        item->ht_item.key = key;
        item->rank = rank;
        item->task_class_id  = task->task_class->task_class_id;
        parsec_hash_table_lock_bucket(migrated_task_ht, key);
        parsec_hash_table_nolock_insert(migrated_task_ht, &item->ht_item);
        parsec_hash_table_unlock_bucket(migrated_task_ht, key);
    }
    return 1;
}

int find_migrated_tasks_details(parsec_task_t *task)
{
    parsec_key_t key;
    mig_task_mapping_item_t *item;

    key = task->task_class->make_key(task->taskpool, task->locals);
    if (NULL == (item = parsec_hash_table_nolock_find(migrated_task_ht, key))) {
        return -1;
    }
    if(task->task_class->task_class_id != item->task_class_id) {
        return -1;
    }

    assert(0 <= item->rank && item->rank < get_nb_nodes());
    return item->rank;
}

int insert_received_tasks_details(parsec_task_t *task, int rank)
{
    parsec_key_t key;
    mig_task_mapping_item_t *item;

    assert(0 <= rank && rank < get_nb_nodes());
    key = task->task_class->make_key(task->taskpool, task->locals);

    /**
     * @brief Entry NULL imples that this task has never been received
     * till now in any of the iteration. So we start a new entry.
     */
    if (NULL == (item = parsec_hash_table_nolock_find(received_task_ht, key)))
    {
        item = (mig_task_mapping_item_t *)malloc(sizeof(mig_task_mapping_item_t));
        item->ht_item.key = key;
        item->rank = rank;
        item->task_class_id = task->task_class->task_class_id;
        parsec_hash_table_lock_bucket(received_task_ht, key);
        parsec_hash_table_nolock_insert(received_task_ht, &item->ht_item);
        parsec_hash_table_unlock_bucket(received_task_ht, key);
    }
    return 1;
}

int find_received_tasks_details(parsec_task_t *task)
{
    parsec_key_t key;
    mig_task_mapping_item_t *item;

    key = task->task_class->make_key(task->taskpool, task->locals);
    if (NULL == (item = parsec_hash_table_nolock_find(received_task_ht, key))) {
        return -1;
    }
    if(task->task_class->task_class_id != item->task_class_id) {
        return -1;
    }

    assert(0 <= item->rank && item->rank < get_nb_nodes());

    return item->rank;
}

int insert_direct_msg(parsec_task_t *task, int rank)
{
    parsec_key_t key;
    mig_task_mapping_item_t *item;

    assert(0 <= rank && rank < get_nb_nodes());
    key = task->task_class->make_key(task->taskpool, task->locals);

    /**
     * @brief Entry NULL imples that this task has never been received
     * till now in any of the iteration. So we start a new entry.
     */
    if (NULL == (item = parsec_hash_table_nolock_find(direct_msg_ht, key)))
    {
        item = (mig_task_mapping_item_t *)malloc(sizeof(mig_task_mapping_item_t));
        item->ht_item.key = key;
        item->rank = rank;
        item->task_class_id = task->task_class->task_class_id;
        parsec_hash_table_lock_bucket(direct_msg_ht, key);
        parsec_hash_table_nolock_insert(direct_msg_ht, &item->ht_item);
        parsec_hash_table_unlock_bucket(direct_msg_ht, key);
    }
    return 1;
}

int find_direct_msg(parsec_task_t *task)
{
    parsec_key_t key;
    mig_task_mapping_item_t *item;

    key = task->task_class->make_key(task->taskpool, task->locals);
    if (NULL == (item = parsec_hash_table_nolock_find(direct_msg_ht, key))) {
        return -1;
    }
    if(task->task_class->task_class_id != item->task_class_id) {
        return -1;
    }

    assert(0 <= item->rank && item->rank < get_nb_nodes());

    return item->rank;
}

int send_task_mapping_info_to_predecessor(parsec_execution_stream_t *es, parsec_task_t *task,
    int rank)
{
    int src_rank = 0;
    int32_t source_mask = task->sources;
    mig_task_mapping_info_t mapping_info;

    mapping_info.key            = task->task_class->make_key(task->taskpool, task->locals);;
    mapping_info.mig_rank       = rank;
    mapping_info.task_class_id  = task->task_class->task_class_id;
    mapping_info.taskpool_id    = task->taskpool->taskpool_id;
    assert(0 <= rank && rank < get_nb_nodes());

    for (src_rank = 0; source_mask >> src_rank; src_rank++) {

        if (!((1U << src_rank) & source_mask)) {
            continue;
        }
        assert(0 <= src_rank && src_rank < get_nb_nodes());

        if(src_rank == my_rank) {
            update_task_mapping(&mapping_info);          
        }
        else {
            send_task_mapping_info(es, task, &mapping_info, src_rank);
        }
    }

    return 0;
}


int update_task_mapping(mig_task_mapping_info_t *mapping_info)
{
    mig_task_mapping_item_t *item;
    parsec_key_t key = mapping_info->key;
    int task_class_id = mapping_info->task_class_id;
    int new_rank = mapping_info->mig_rank;

    assert(0 <= mapping_info->mig_rank && mapping_info->mig_rank < get_nb_nodes());

    /**
     * @brief Entry NULL imples that this task has never been migrated
     * till now in any of the iteration. So we start a new entry.
     */
    if (NULL == (item = parsec_hash_table_nolock_find(task_map_ht, key)))
    {

        item = (mig_task_mapping_item_t *)malloc(sizeof(mig_task_mapping_item_t));
        item->rank = new_rank;
        item->ht_item.key = key;
        item->task_class_id = task_class_id;
        parsec_hash_table_lock_bucket(task_map_ht, key);
        parsec_hash_table_nolock_insert(task_map_ht, &item->ht_item);
        parsec_hash_table_unlock_bucket(task_map_ht, key);

        return item->rank;
    }
    else {
        if(item->task_class_id != task_class_id) {
            return -1;
        }
        item->rank = new_rank;
        item->ht_item.key = key;
        item->task_class_id = task_class_id;
        return item->rank;
    }

    return -1;
}

int find_task_mapping(parsec_task_t *task)
{
    parsec_key_t key;
    mig_task_mapping_item_t *item;

    assert(NULL != task->taskpool);

    key = task->task_class->make_key(task->taskpool, task->locals);
    if (NULL == (item = parsec_hash_table_nolock_find(task_map_ht, key))) {
        return -1;
    }

    if(task->task_class->task_class_id != item->task_class_id) {
        return -1;
    }

    assert(0 <= item->rank && item->rank < get_nb_nodes());
    return item->rank;
}

int send_task_mapping_info(parsec_execution_stream_t *eu, const parsec_task_t *task,
    mig_task_mapping_info_t *mapping_info, int src)
                                 
{
    task->taskpool->tdm.module->outgoing_message_start(task->taskpool, mapping_info->mig_rank, NULL);
    parsec_ce.send_am(&parsec_ce, PARSEC_MIG_INFORM_PREDECESSOR_TAG, src, mapping_info, MAPPING_INFO_SIZE);

    return 0;
}

static int
update_task_mapping_cb(parsec_comm_engine_t *ce, parsec_ce_tag_t tag,
    void *msg, size_t msg_size, int src, void *cb_data)
{
    mig_task_mapping_info_t *mapping_info = (mig_task_mapping_info_t *)msg;
    assert( MAPPING_INFO_SIZE == msg_size );
    parsec_taskpool_t *taskpool = parsec_taskpool_lookup(mapping_info->taskpool_id);

    taskpool->tdm.module->incoming_message_start(taskpool, src, msg, NULL, 0, NULL);
    assert(0 <= mapping_info->mig_rank && mapping_info->mig_rank < get_nb_nodes());

    update_task_mapping(mapping_info);
    taskpool->tdm.module->incoming_message_end(taskpool, NULL);
    
    return 0;
}


/**
 * Starting with a particular item pack as many remote_dep_wire_activate
 * messages with the same destination (from the item ring associated with
 * pos_list) into a buffer. Upon completion the entire buffer is send to the
 * remote peer, the completed messages are released and the header is updated to
 * the next unsent message.
 */
int mig_dep_direct_send(parsec_execution_stream_t* es, int rank, parsec_remote_deps_t *deps)
{
    (void)es;
    char packed_buffer[SINGLE_ACTIVATE_MSG_SIZE];
    int  position = 0, saved_position = 0, dsize = 0;

    /** Size of the data to be send  */
    parsec_ce.pack_size(&parsec_ce, SINGLE_ACTIVATE_MSG_SIZE, parsec_datatype_int8_t, &dsize);
    /** one task details  size. It is important to set this here, so as to be used on the receiver */
    deps->msg.length = dsize; 

    /** pack the message to the buffer starting at position*/
    parsec_ce.pack(&parsec_ce, &deps->msg, SINGLE_ACTIVATE_MSG_SIZE, parsec_datatype_int8_t, packed_buffer, SINGLE_ACTIVATE_MSG_SIZE, &saved_position);
    /** Update the position. Next task details will start from this position*/
    position = position + dsize;

  
    parsec_ce.send_am(&parsec_ce, PARSEC_MIG_DEP_DIRECT_ACTIVATE_TAG, rank, packed_buffer, position);
   
    return 0;
    
}

static int
mig_direct_activate_cb(parsec_comm_engine_t *ce, parsec_ce_tag_t tag,
    void *msg, size_t msg_size, int src, void *cb_data)
{
    (void) tag; (void) cb_data;
    parsec_execution_stream_t* es = &parsec_comm_es;

    int position = 0, saved_position = 0, length = msg_size, rc;
    parsec_remote_deps_t* deps = NULL;

    while(position < length) {
        deps = remote_deps_allocate(&parsec_remote_dep_context.freelist);

        saved_position = position;
        parsec_ce.unpack(&parsec_ce, msg, length, &position, &deps->msg, SINGLE_ACTIVATE_MSG_SIZE, parsec_datatype_int8_t);
        deps->from = src;
        deps->eager_msg = msg;
        deps->root = deps->msg.root;

        /* Retrieve the data arenas and update the msg.incoming_mask to reflect
         * the data we should be receiving from the predecessor.
         */
        rc = mig_direct_get_datatypes(es, deps, 0, &position);
        if( -1 == rc ) {
            char* packed_buffer;
            packed_buffer = malloc(deps->msg.length);
            memcpy(packed_buffer, msg + saved_position, deps->msg.length);
            deps->taskpool = (parsec_taskpool_t*)packed_buffer;  /* temporary storage */
            parsec_list_push_back(&direct_msg_fifo, (parsec_list_item_t*)deps);
        } 
        rc = check_deps_received(es, deps);
        if ( -2 == rc) {
            /** The same message was received from someone else */
            mig_direct_no_get_start(es, deps);
            free(deps);
        }
        else {
            mig_direct_recv_activate(es, deps, msg,
                                     saved_position + deps->msg.length, &position);
        }
    }
    assert(position == length);

    return 1;
}

static int
mig_direct_get_datatypes(parsec_execution_stream_t* es,
                         parsec_remote_deps_t* origin,
                         int storage_id, int *position)
{
    uint32_t i, j, k, local_mask = 0, rc = -1;

    assert(NULL == origin->taskpool);
    origin->taskpool = parsec_taskpool_lookup(origin->msg.taskpool_id);
    if( NULL == origin->taskpool )
        return -1; /* the parsec taskpool doesn't exist yet */

    /* This function is divided into DTD and PTG's logic */
    if( PARSEC_TASKPOOL_TYPE_PTG == origin->taskpool->taskpool_type ) {
        parsec_task_t task;
        task.taskpool   = origin->taskpool;
        /* Do not set the task.task_class here, because it might trigger a race condition in DTD */

        task.priority = 0;  /* unknown yet */

        task.task_class = task.taskpool->task_classes_array[origin->msg.task_class_id];
        for(i = 0; i < task.task_class->nb_flows;
            task.data[i].data_in = task.data[i].data_out = NULL,
            task.data[i].source_repo_entry = NULL,
            task.data[i].source_repo = NULL, i++);

        for(i = 0; i < task.task_class->nb_locals; i++)
            task.locals[i] = origin->msg.locals[i];

        /* We need to convert from a dep_datatype_index mask into a dep_index
         * mask. However, in order to be able to use the above iterator we need to
         * be able to identify the dep_index for each particular datatype index, and
         * call the iterate_successors on each of the dep_index sets.
         */
        for(k = 0; origin->msg.output_mask>>k; k++) {
            if(!(origin->msg.output_mask & (1U<<k))) continue;
            for(local_mask = i = 0; NULL != task.task_class->out[i]; i++ ) {
                if(!(task.task_class->out[i]->flow_datatype_mask & (1U<<k))) continue;
                for(j = 0; NULL != task.task_class->out[i]->dep_out[j]; j++ )
                    if(k == task.task_class->out[i]->dep_out[j]->dep_datatype_index)
                        local_mask |= (1U << task.task_class->out[i]->dep_out[j]->dep_index);
                if( 0 != local_mask ) break;  /* we have our local mask, go get the datatype */
            }

            origin->output[k].data.remote.src_datatype = origin->output[k].data.remote.dst_datatype = PARSEC_DATATYPE_NULL;
            PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "MPI:\tRetrieve datatype with mask 0x%x (remote_dep_get_datatypes)", local_mask);
            task.task_class->iterate_successors(es, &task,
                                                local_mask,
                                                mig_dep_mpi_retrieve_datatype,
                                                origin);
        }
    }
    else
    {
        assert(0);
    }

    /**
     * At this point the msg->output_mask contains the root mask, and should be
     * keep as is and be propagated down the communication pattern. On the
     * origin->incoming_mask we have the mask of all local data to be retrieved
     * from the predecessor.
     */
    origin->outgoing_mask = origin->incoming_mask;  /* safekeeper */
    return 0;
}

parsec_ontask_iterate_t
mig_dep_mpi_retrieve_datatype(parsec_execution_stream_t *eu,
    const parsec_task_t *newcontext, const parsec_task_t *oldcontext,
    const parsec_dep_t* dep, parsec_dep_data_description_t* out_data,
    int src_rank, int dst_rank, int dst_vpid, data_repo_t *successor_repo, 
    parsec_key_t successor_repo_key, void *param)
{
    (void)eu; (void)oldcontext; (void)dst_vpid; (void)newcontext; (void)out_data;
    (void)successor_repo; (void) successor_repo_key;

    int was_received = find_received_tasks_details(newcontext);

    /** we are only intrested in task that were migrated in the previous iteeration */
    if( -1 ==  was_received) {
        return PARSEC_ITERATE_CONTINUE;
    }

    parsec_remote_deps_t *deps               = (parsec_remote_deps_t*)param;
    struct remote_dep_output_param_s* output = &deps->output[dep->dep_datatype_index];
    const parsec_task_class_t* fct           = newcontext->task_class;
    uint32_t flow_mask                       = (1U << dep->flow->flow_index) | 0x80000000;  /* in flow */

    parsec_datatype_t old_dtt = output->data.remote.dst_datatype;

    /* Extract the datatype, count and displacement from the target task */

    fct->get_datatype(eu, newcontext, &flow_mask, &output->data);
    

    parsec_data_t* data_arena =  NULL; //is_read_only(oldcontext, dep);
    if(NULL == data_arena) {
        output->deps_mask &= ~(1U << dep->dep_index); /* unmark all data that are RO we already hold from previous tasks */
    } else {
        output->deps_mask |= (1U << dep->dep_index); /* mark all data that are not RO */
        data_arena = NULL; //is_inplace(oldcontext, dep);  /* Can we do it inplace */
    }
    output->data.data = NULL;

    if( deps->max_priority < newcontext->priority ) deps->max_priority = newcontext->priority;
    deps->incoming_mask |= (1U << dep->dep_datatype_index);

    if(output->data.remote.dst_count == 0){
        /* control dep */
        return PARSEC_ITERATE_STOP;
    }
    if(old_dtt != PARSEC_DATATYPE_NULL){
        if(old_dtt != output->data.remote.dst_datatype){
            // TODO JS: implement MPI_Pack_size
            int dsize;
            MPI_Pack_size(output->data.remote.dst_count, output->data.remote.dst_datatype, MPI_COMM_WORLD, &dsize);
            output->data.remote.src_count = output->data.remote.dst_count = dsize;
            output->data.remote.src_datatype = output->data.remote.dst_datatype = PARSEC_DATATYPE_PACKED;

            return PARSEC_ITERATE_STOP;
        }
    }

    return PARSEC_ITERATE_CONTINUE;
}

static void mig_direct_recv_activate(parsec_execution_stream_t* es,
                                         parsec_remote_deps_t* deps,
                                         char* packed_buffer,
                                         int length,
                                         int* position)
{
    (void) length; (void) position;
    (void) packed_buffer;
    remote_dep_datakey_t complete_mask = 0;
    int k;

    deps->taskpool->tdm.module->incoming_message_start(deps->taskpool, deps->from, packed_buffer, position,
                                                       length, deps);
        
    for(k = 0; deps->incoming_mask>>k; k++) {
        if(!(deps->incoming_mask & (1U<<k))) continue;
        /* Check for CTL and data that do not carry payload */
        if( parsec_is_CTL_dep(deps->output[k].data) ){
            /* deps->output[k].data.data = NULL; This is unnecessary*/
            complete_mask |= (1U<<k);
            continue;
        }
    }
    assert(length == *position);

    /* Release all the already satisfied deps without posting the RDV */
    if(complete_mask) {
        /* If this is the only call then force the remote deps propagation */
        deps = mig_direct_release_incoming(es, deps, complete_mask);
    }

    ///* Store the request in the rdv queue if any unsatisfied dep exist at this point */
    //if(NULL != deps) {
    //    assert(0 != deps->incoming_mask);
    //    assert(0 != deps->msg.output_mask);
    //    parsec_list_nolock_push_sorted(&dep_activates_fifo, (parsec_list_item_t*)deps, rdep_prio);
    //}

    ///* Check if we have any pending GET orders */
    //if(parsec_ce.can_serve(&parsec_ce) && !parsec_list_nolock_is_empty(&dep_activates_fifo)) {
    //    deps = (parsec_remote_deps_t*)parsec_list_nolock_pop_front(&dep_activates_fifo);
    //    remote_dep_mpi_get_start(es, deps);
    //}

    mig_direct_get_start(es, deps);
}

static void mig_direct_get_start(parsec_execution_stream_t* es,
                                     parsec_remote_deps_t* deps)
{
    remote_dep_wire_activate_t* task = &(deps->msg);
    int from = deps->from, k, count, nbdtt;
    remote_dep_wire_get_t msg;
    MPI_Datatype dtt;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN], type_name[MPI_MAX_OBJECT_NAME];
    int len;
    remote_dep_cmd_to_string(task, tmp, MAX_TASK_STRLEN);
#endif
#ifdef PARSEC_PROF_TRACE
    int32_t save_get = 0;
    int32_t event_id = parsec_atomic_fetch_inc_int32(&save_get);
#endif  /* PARSEC_PROF_TRACE */
    for(k = count = 0; deps->incoming_mask >> k; k++)
        if( ((1U<<k) & deps->incoming_mask) ) count++;

    (void)es;

    msg.source_deps = task->deps; /* the deps copied from activate message from source */
    msg.callback_fn = (uintptr_t)mig_direct_get_end_cb; /* We let the source know to call this
                                                             * function when the PUT is over, in a true
                                                             * one sided case the (integer) value of this
                                                             * function pointer will be registered as the
                                                             * TAG to receive the same notification. */

    for(k = 0; deps->incoming_mask >> k; k++) {
        if( !((1U<<k) & deps->incoming_mask) ) continue;
        msg.output_mask = 0;  /* Only get what I need */
        msg.output_mask |= (1U<<k);

        /* We pack the callback data that should be passed to us when the other side
         * notifies us to invoke the callback_fn we have assigned above
         */
        remote_dep_cb_data_t *callback_data = (remote_dep_cb_data_t *) parsec_thread_mempool_allocate
                                                    (parsec_remote_dep_cb_data_mempool->thread_mempools);
        callback_data->deps = deps;
        callback_data->k    = k;

        /* prepare the local receiving data */
        assert(NULL == deps->output[k].data.data); /* we do not support in-place tiles now, make sure it doesn't happen yet */
        if(NULL == deps->output[k].data.data) {
            deps->output[k].data.data = migrated_copy_allocate(&deps->output[k].data.remote);
        }
        dtt   = deps->output[k].data.remote.dst_datatype;
        nbdtt = deps->output[k].data.remote.dst_count;

        /* We have the remote mem_handle.
         * Let's allocate our mem_reg_handle
         * and let the source know.
         */
        parsec_ce_mem_reg_handle_t receiver_memory_handle;
        size_t receiver_memory_handle_size;

        if(parsec_ce.capabilites.supports_noncontiguous_datatype) {
            parsec_ce.mem_register(PARSEC_DATA_COPY_GET_PTR(deps->output[k].data.data), PARSEC_MEM_TYPE_NONCONTIGUOUS,
                                   nbdtt, dtt,
                                   -1,
                                   &receiver_memory_handle, &receiver_memory_handle_size);
        } else {
            /* TODO: Implement converter to pack and unpack */
            int dtt_size;
            parsec_type_size(dtt, &dtt_size);
            parsec_ce.mem_register(PARSEC_DATA_COPY_GET_PTR(deps->output[k].data.data), PARSEC_MEM_TYPE_CONTIGUOUS,
                                   -1, NULL,
                                   dtt_size,
                                   &receiver_memory_handle, &receiver_memory_handle_size);

        }

#  if defined(PARSEC_DEBUG_NOISIER)
        MPI_Type_get_name(dtt, type_name, &len);
        int _size;
        MPI_Type_size(dtt, &_size);
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "MPI:\tTO\t%d\tGet START\t% -8s\tk=%d\twith datakey %lx at %p type %s count %d displ %ld \t(k=%d, dst_mem_handle=%p)",
                from, tmp, k, task->deps, PARSEC_DATA_COPY_GET_PTR(deps->output[k].data.data), type_name, nbdtt,
                deps->output[k].data.remote.dst_displ, k, receiver_memory_handle);
#  endif

        callback_data->memory_handle = receiver_memory_handle;

        /* We need multiple information to be passed to the callback_fn we have assigned above.
         * We pack the pointer to this callback_data and pass to the other side so we can complete
         * cleanup and take necessary action when the data is available on our side */
        msg.remote_callback_data = (remote_dep_datakey_t)callback_data;

        /* We pack the static message(remote_dep_wire_get_t) and our memory_handle and send this message
         * to the source. Source is anticipating this exact configuration.
         */
        int buf_size = sizeof(remote_dep_wire_get_t) + receiver_memory_handle_size;
        void *buf = malloc(buf_size);
        memcpy( buf,
                &msg,
                sizeof(remote_dep_wire_get_t) );
        memcpy( ((char*)buf) +  sizeof(remote_dep_wire_get_t),
                receiver_memory_handle,
                receiver_memory_handle_size );

        /* Send AM */
        parsec_ce.send_am(&parsec_ce, PARSEC_CE_REMOTE_DEP_GET_DATA_TAG, from, buf, buf_size);
        TAKE_TIME(es->es_profile, MPI_Data_ctl_ek, event_id);

        free(buf);

        parsec_comm_gets++;
    }
}

static int
mig_direct_get_end_cb(parsec_comm_engine_t *ce,
                          parsec_ce_tag_t tag,
                          void *msg,
                          size_t msg_size,
                          int src,
                          void *cb_data)
{
    (void) ce; (void) tag; (void) msg_size; (void) cb_data; (void) src;
    parsec_execution_stream_t* es = &parsec_comm_es;

    /* We send 8 bytes to the source to give it back to us when the PUT is completed,
     * let's retrieve that
     */
    uintptr_t *retrieve_pointer_to_callback = (uintptr_t *)msg;
    remote_dep_cb_data_t *callback_data = (remote_dep_cb_data_t *)*retrieve_pointer_to_callback;
    parsec_remote_deps_t *deps = (parsec_remote_deps_t *)callback_data->deps;

#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif

    PARSEC_DEBUG_VERBOSE(6, parsec_debug_output, "MPI:\tFROM\t%d\tGet END  \t% -8s\tk=%d\twith datakey na        \tparams %lx\t(tag=%d)",
            src, remote_dep_cmd_to_string(&deps->msg, tmp, MAX_TASK_STRLEN),
            callback_data->k, deps->incoming_mask, src);


    TAKE_TIME(es->es_profile, MPI_Data_pldr_ek, callback_data->k);
    mig_direct_get_end(es, callback_data->k, deps);

    parsec_ce.mem_unregister(&callback_data->memory_handle);
    parsec_thread_mempool_free(parsec_remote_dep_cb_data_mempool->thread_mempools, callback_data);

    parsec_comm_gets--;

    return 1;
}

static void mig_direct_get_end(parsec_execution_stream_t* es,
                                   int idx,
                                   parsec_remote_deps_t* deps)
{
    /* The ref on the data will be released below */
    mig_direct_release_incoming(es, deps, (1U<<idx));
}

static parsec_remote_deps_t*
mig_direct_release_incoming(parsec_execution_stream_t* es,
                            parsec_remote_deps_t* origin,
                            remote_dep_datakey_t complete_mask)
{
    parsec_task_t task;
    const parsec_flow_t* target;
    int i, pidx;
    uint32_t action_mask = 0;

    /* Update the mask of remaining dependencies to avoid releasing the same outputs twice */
    assert((origin->incoming_mask & complete_mask) == complete_mask);
    origin->incoming_mask ^= complete_mask;

    task.taskpool = origin->taskpool;
    task.task_class = task.taskpool->task_classes_array[origin->msg.task_class_id];
    task.priority = origin->priority;
    for(i = 0; i < task.task_class->nb_locals;
        task.locals[i] = origin->msg.locals[i], i++);
    for(i = 0; i < task.task_class->nb_flows;
        task.data[i].data_in = task.data[i].data_out = NULL, task.data[i].source_repo_entry = NULL, task.data[i].source_repo = NULL, i++);
    task.repo_entry = NULL;

    for(i = 0; complete_mask>>i; i++) {
        assert(i < MAX_PARAM_COUNT);
        if( !((1U<<i) & complete_mask) ) continue;
        pidx = 0;
        target = task.task_class->out[pidx];
        while( !((1U<<i) & target->flow_datatype_mask) ) {
            target = task.task_class->out[++pidx];
            assert(NULL != target);
        }
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "MPI:\tDATA %p(%s) released from %p[%d] flow idx %d",
                origin->output[i].data.data, target->name, origin, i, target->flow_index);
        task.data[target->flow_index].source_repo = NULL;
        task.data[target->flow_index].source_repo_entry = NULL;
        task.data[target->flow_index].data_in   = origin->output[i].data.data;
        task.data[target->flow_index].data_out  = origin->output[i].data.data;
    }


    if(PARSEC_TASKPOOL_TYPE_PTG == origin->taskpool->taskpool_type) {
        /* We need to convert from a dep_datatype_index mask into a dep_index mask */
        for(int i = 0; NULL != task.task_class->out[i]; i++ ) {
            target = task.task_class->out[i];
            if( !(complete_mask & target->flow_datatype_mask) ) continue;
            for(int j = 0; NULL != target->dep_out[j]; j++ )
                if(complete_mask & (1U << target->dep_out[j]->dep_datatype_index))
                    action_mask |= (1U << target->dep_out[j]->dep_index);
        }
    } else {
        assert(0);
    }
    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "MPI:\tTranslate mask from 0x%lx to 0x%x (remote_dep_release_incoming)",
            complete_mask, action_mask);
    (void)task.task_class->release_deps(es, &task,
                                        action_mask | PARSEC_ACTION_RELEASE_DIRECT_DEPS | PARSEC_ACTION_RESHAPE_REMOTE_ON_RELEASE,
                                        NULL);
    assert(0 == (origin->incoming_mask & complete_mask));

    if(0 != origin->incoming_mask)  /* not done receiving */
        return origin;

    origin->taskpool->tdm.module->incoming_message_end(origin->taskpool, origin);
    
    //uint32_t mask = origin->outgoing_mask;
    ///**
    // * Release the dependency owned by the communication engine for all data
    // * internally allocated by the engine.
    // */
    //for(i = 0; mask>>i; i++) {
    //    assert(i < MAX_PARAM_COUNT);
    //    if( !((1U<<i) & mask) ) continue;
    //    if( NULL != origin->output[i].data.data )  /* except CONTROLs */
    //    {
    //        PARSEC_DATA_COPY_RELEASE(origin->output[i].data.data);
    //    }
    //}

    //remote_deps_free(origin);

    return NULL;
}

void remote_dep_mark_forwarded_direct(parsec_execution_stream_t* es,
    parsec_remote_deps_t* rdeps, int rank)
{
    uint32_t boffset;

    boffset = rank / (8 * sizeof(uint32_t));
    assert(boffset <= es->virtual_process->parsec_context->remote_dep_fw_mask_sizeof);
    (void)es;
    rdeps->remote_dep_fw_mask_direct[boffset] |= ((uint32_t)1) << (rank % (8 * sizeof(uint32_t)));
}

int remote_dep_is_forwarded_direct(parsec_execution_stream_t* es,
    parsec_remote_deps_t* rdeps, int rank)
{
    uint32_t boffset, mask;

    boffset = rank / (8 * sizeof(uint32_t));
    mask = ((uint32_t)1) << (rank % (8 * sizeof(uint32_t)));
    PARSEC_DEBUG_VERBOSE(20, parsec_comm_output_stream, "fw test\tREMOTE rank %d (value=%x)", rank, (int) (rdeps->remote_dep_fw_mask[boffset] & mask));
    (void)es;
    return (int) ((rdeps->remote_dep_fw_mask_direct[boffset] & mask) != 0);
}

int change_destination(parsec_execution_stream_t *es, parsec_release_dep_fct_arg_t *arg, 
    parsec_task_t task, int src, int dst)
{
    
}

static int mig_no_put_cb(parsec_comm_engine_t *ce, parsec_ce_tag_t tag, void *msg,
    size_t msg_size, int src, void *cb_data)
{
    (void) ce; (void) tag; (void) cb_data; (void) msg_size;
    remote_dep_wire_get_t* task;
    parsec_remote_deps_t *deps;

    parsec_execution_stream_t* es = &parsec_comm_es;
    task = malloc(sizeof(remote_dep_wire_get_t));

    /** copy the static message */
    memcpy(task, msg, sizeof(remote_dep_wire_get_t));

    /* we are expecting exactly one wire_get_t + remote memory handle */
    assert(msg_size == sizeof(remote_dep_wire_get_t));

    int total_cleanup = task->output_mask; /** Get the total_cleanup from the temprary storage */
    deps = (parsec_remote_deps_t*)(remote_dep_datakey_t)task->source_deps; /* get our deps back */

    remote_dep_complete_and_cleanup(&deps, total_cleanup);
    free(task);
    
    return 1;
}

static void mig_direct_no_get_start(parsec_execution_stream_t* es,
                                     parsec_remote_deps_t* deps)
{
    remote_dep_wire_activate_t* task = &(deps->msg);
    int from = deps->from, k, position = 0;
    remote_dep_wire_get_t msg;

    assert(0 <= from && from < get_nb_nodes());

    (void)es;
    int total_cleanup = 0;

    deps->taskpool->tdm.module->incoming_message_start(deps->taskpool, deps->from, &deps->msg, 
        &position, deps->msg.length, deps);
                
    msg.source_deps = task->deps; /* the deps copied from activate message from source */
    
    for(k = 0; deps->incoming_mask >> k; k++) {
        if( !((1U<<k) & deps->incoming_mask) ) continue;

        total_cleanup++;
    }
    msg.output_mask = total_cleanup; /** temporary storage for the source */

    /** We pack the static message(remote_dep_wire_get_t) and the total_cleanup required
     *  on the source side, to the source. 
     */
    int buf_size = sizeof(remote_dep_wire_get_t);
    void *buf = malloc(buf_size);
    memcpy( buf, &msg, sizeof(remote_dep_wire_get_t) );

    parsec_ce.send_am(&parsec_ce, PARSEC_MIG_NO_GET_DATA_TAG, from, buf, buf_size);

    deps->taskpool->tdm.module->incoming_message_end(deps->taskpool, deps);
    free(buf);
}


static int check_deps_received(parsec_execution_stream_t* es, parsec_remote_deps_t* origin)
{
    uint32_t i, j, k, local_mask = 0, rc = -1;

    assert(NULL != origin->taskpool);

    /* This function is divided into DTD and PTG's logic */
    if( PARSEC_TASKPOOL_TYPE_PTG == origin->taskpool->taskpool_type ) {
        parsec_task_t task;
        task.taskpool   = origin->taskpool;
        /* Do not set the task.task_class here, because it might trigger a race condition in DTD */

        task.priority = 0;  /* unknown yet */

        task.task_class = task.taskpool->task_classes_array[origin->msg.task_class_id];
        for(i = 0; i < task.task_class->nb_flows;
            task.data[i].data_in = task.data[i].data_out = NULL,
            task.data[i].source_repo_entry = NULL,
            task.data[i].source_repo = NULL, i++);

        for(i = 0; i < task.task_class->nb_locals; i++)
            task.locals[i] = origin->msg.locals[i];

        rc = find_direct_msg(&task);
        /** check if this message was already received from someone else */
        if(-1 != rc) {
            assert(0 <= rc && rc < get_nb_nodes());
            return -2;
        }
        insert_direct_msg(&task, origin->from);
        assert(origin->from != rc);
    }
    else {
        assert(0);
    }
    
    return 1;
}


