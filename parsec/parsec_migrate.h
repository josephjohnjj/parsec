#ifndef MIGRATE_H
#define MIGRATE_H

#include "parsec/class/dequeue.h"
#include "parsec/class/lifo.h"
#include "parsec/parsec_internal.h"
#include "parsec/scheduling.h"
#include "parsec/mca/sched/sched.h"
#include "parsec/parsec_comm_engine.h"
#include "parsec/remote_dep.h"
#include <pthread.h> 
#include "parsec/parsec_comm_engine.h"

#define SINGLE_ACTIVATE_MSG_SIZE sizeof(remote_dep_wire_activate_t)
#define MAX_CHUNK_SIZE           20
#define MAX_ACTIVATE_MSG_SIZE    (MAX_CHUNK_SIZE * SINGLE_ACTIVATE_MSG_SIZE)
#define STEAL_REQ_MSG_SIZE       sizeof(steal_request_msg_t)
#define MAPPING_INFO_SIZE        sizeof(mig_task_mapping_info_t)
#define NB_TASK_CLASS_HTS        16
#define ACK_MSG_SIZE             sizeof(mig_ack_msg_t)

#define PARSEC_NON_MIGRATED_TASK    (uint8_t)0x00
#define PARSEC_MIGRATED_TASK        (uint8_t)0x01 
#define PARSEC_MIGRATED_DIRECT      (uint8_t)0x02 
#define PARSEC_MIGRATED_TEST        (uint8_t)0x03 

#define MAX_NODES           ((uint32_t)1024)
#define RANKS_PER_INDEX     (sizeof(uint32_t))
#define MAX_NODES_INDEX     (MAX_NODES/RANKS_PER_INDEX)



typedef struct steal_request_msg_s{
    int root;
    int src;
    int dst;
    int nb_task_request;
    int hop_count;
    uint32_t successful_victims[MAX_NODES_INDEX];
    uint32_t failed_victims[MAX_NODES_INDEX];
} steal_request_msg_t;

typedef struct steal_request_s{
    parsec_list_item_t  super;
    steal_request_msg_t msg;
} steal_request_t;

typedef struct parsec_node_info_s 
{
    int nb_gpu_tasks_executed;
    int nb_cpu_tasks_executed;
    int nb_req_send;
    int nb_req_recvd;
    int nb_req_forwarded;
    int nb_task_migrated;
    int nb_task_recvd;
    int nb_req_processed;
    int nb_succesfull_req;
    int nb_searches;
    int full_yield;
    int nb_release;
    int nb_selected;
    int nb_succesfull_req_processed;
    int nb_succesfull_steals;
    int nb_succesfull_full_steals;
    int hops_succesfull_steals;
    int hops_succesfull_full_steals;
    
} parsec_node_info_t;

typedef struct node_prof_s 
{
    double ready_tasks;
    double complete_time;
} node_prof_t;

typedef struct steal_req_prof_s 
{
    double gpu_tasks;
    double recv_time;
} steal_req_prof_t;

typedef struct migrated_node_level_task_s
{
    parsec_list_item_t list_item;
    parsec_task_t *task;
    int root;
} migrated_node_level_task_t;

typedef struct mig_task_mapping_item_s
{
    parsec_hash_table_item_t ht_item;
    int thief;
    int victim;
    int task_class_id;
} mig_task_mapping_item_t;

typedef struct mig_task_mapping_info_s
{
    parsec_key_t key;
    int taskpool_id;
    int thief;
    int victim;
    int task_class_id;
} mig_task_mapping_info_t;

typedef struct mig_ack_msg_s {
    uint32_t             taskpool_id;
    uint32_t             task_class_id;
    int                  victim;
    int                  thief;
    parsec_assignment_t  locals[MAX_LOCAL_COUNT];
} mig_ack_msg_t;

int parsec_node_migrate_init(parsec_context_t* context );
int parsec_node_migrate_fini();
int send_steal_request(parsec_execution_stream_t* es);
int process_steal_request(parsec_execution_stream_t* es);
int process_mig_request(parsec_task_t* this_task);
int parsec_node_stats_init();
int parsec_node_stats_fini();
int parsec_node_mig_inc_selected();
int parsec_node_mig_inc_gpu_task_executed();
int parsec_node_mig_inc_cpu_task_executed();
void mig_new_taskpool(parsec_execution_stream_t* es, dep_cmd_item_t *dep_cmd_item);
int migrate_put_mpi_progress(parsec_execution_stream_t* es);
int nb_starving_device(parsec_execution_stream_t *es);
int progress_migrated_task(parsec_execution_stream_t* es);
int increment_progress_counter(int device_num);
int unset_progress_counter(int device_num);
int get_progress_counter(int device_num);
parsec_dependency_t parsec_update_sources(const parsec_taskpool_t *tp, parsec_execution_stream_t *es,
    const parsec_task_t* restrict task, parsec_release_dep_fct_arg_t *arg, int src_rank);
int get_nb_nodes();
int get_my_rank();
mig_task_mapping_item_t* find_task_mapping(const parsec_task_t *task);
int remote_dep_is_forwarded_direct(parsec_execution_stream_t* es,
    parsec_remote_deps_t* rdeps, int rank);
void remote_dep_mark_forwarded_direct(parsec_execution_stream_t* es,
    parsec_remote_deps_t* rdeps, int rank);
void remote_dep_reset_forwarded_direct(parsec_execution_stream_t* es,
    parsec_remote_deps_t* rdeps);
mig_task_mapping_item_t* find_received_tasks_details(const parsec_task_t *task);
mig_task_mapping_item_t* insert_received_tasks_details(parsec_task_t *task, int victim, int thief);
mig_task_mapping_item_t* insert_direct_msg(parsec_task_t *task, int current_source);
mig_task_mapping_item_t* find_direct_msg(parsec_task_t *task);
int whoami();
int progress_direct_activation(parsec_execution_stream_t* es);
int direct_activation_fifo_status(parsec_execution_stream_t* es);
int modify_action_for_no_new_mapping(const parsec_task_t *predecessor, const parsec_task_t *succecessor,
    int* src_rank, int* dst_rank, mig_task_mapping_item_t *new_mapping, parsec_release_dep_fct_arg_t *arg);
int modify_action_for_new_mapping(const parsec_task_t *predecessor, const parsec_task_t *succecessor,
    int* src_rank, int* dst_rank, mig_task_mapping_item_t *new_mapping, parsec_release_dep_fct_arg_t *arg);
parsec_ontask_iterate_t parsec_gather_direct_collective_pattern(parsec_execution_stream_t *es,
    const parsec_task_t *newcontext, const parsec_task_t *oldcontext,
    const parsec_dep_t* dep, parsec_dep_data_description_t* data,
    int src_rank, int dst_rank, int dst_vpid,data_repo_t *successor_repo, 
    parsec_key_t successor_repo_key, void *param);
parsec_ontask_iterate_t  mig_dep_mpi_retrieve_datatype(parsec_execution_stream_t *eu,
    const parsec_task_t *newcontext, const parsec_task_t *oldcontext,
    const parsec_dep_t* dep, parsec_dep_data_description_t* out_data,
    int src_rank, int dst_rank, int dst_vpid, data_repo_t *successor_repo, 
    parsec_key_t successor_repo_key, void *param);
mig_task_mapping_item_t* find_migrated_tasks_details(const parsec_task_t *task);
int create_direct_message_ht(parsec_taskpool_t* tp);
int destroy_direct_message_ht(parsec_taskpool_t* tp);
int destroy_task_class_hashtables(parsec_context_t *context);
int create_task_class_hashtables(parsec_context_t *context);
parsec_ontask_iterate_t elastic_test(parsec_execution_stream_t *eu,
    const parsec_task_t *newcontext, const parsec_task_t *oldcontext,
    const parsec_dep_t* dep, parsec_dep_data_description_t* out_data,
    int src_rank, int dst_rank, int dst_vpid, data_repo_t *successor_repo, 
    parsec_key_t successor_repo_key, void *param);
int mig_dep_direct_send(parsec_execution_stream_t* es, dep_cmd_item_t **head_item);
mig_task_mapping_item_t* insert_direct_deps_details(parsec_task_t *task);
mig_task_mapping_item_t* find_direct_deps_details(const parsec_task_t *task);
int get_current_taskpool_id();
int set_current_taskpool_id(int id);
int destroy_possible_memory_leak(parsec_taskpool_t* tp);
int migrate_register_tags();
int migrate_unregister_tags();
#endif