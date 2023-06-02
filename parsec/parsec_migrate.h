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


#define PARSEC_NON_MIGRATED_TASK    (uint8_t)0x00
#define PARSEC_MIGRATED_TASK        (uint8_t)0x01 

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

int parsec_node_migrate_init(parsec_context_t* context );
int parsec_node_migrate_fini();
int send_steal_request(parsec_execution_stream_t* es);
int process_steal_request(parsec_execution_stream_t* es);
int process_mig_request(parsec_task_t* this_task);
int parsec_node_stats_init();
int parsec_node_stats_fini();
int parsec_node_mig_inc_released();
int parsec_node_mig_inc_selected();
int parsec_node_mig_inc_gpu_task_executed();
int parsec_node_mig_inc_cpu_task_executed();
int print_stats();
void mig_new_taskpool(parsec_execution_stream_t* es, dep_cmd_item_t *dep_cmd_item);
int migrate_put_mpi_progress(parsec_execution_stream_t* es);
int nb_starving_device(parsec_execution_stream_t *es);
int progress_migrated_task(parsec_execution_stream_t* es);
int increment_progress_counter(int device_num);
int unset_progress_counter(int device_num);
int get_progress_counter(int device_num);
#endif