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

#define dep_count sizeof(remote_dep_wire_activate_t)
#define dep_dtt parsec_datatype_int8_t

#define PARSEC_NON_MIGRATED_TASK    (uint8_t)0x00
#define PARSEC_MIGRATED_TASK        (uint8_t)0x01 


typedef struct steal_request_s{
    parsec_list_item_t super;
    int root;
    int src;
    int dst;
    int nb_task_request;
} steal_request_t;

typedef struct deps_details_renamee_s
{
    int key;
    int deps_mask;
} deps_details_renamee_t;

typedef struct parsec_node_info_s 
{
    int nb_tasks_executed;
    int nb_req_send;
    int nb_req_recvd;
    int nb_req_forwarded;
    int nb_task_migrated;
    int nb_task_recvd;
    int nb_req_processed;
    int nb_succesfull_req;
    int nb_searches;
    
} parsec_node_info_t;

int parsec_node_migrate_init(parsec_context_t* context );
int parsec_node_migrate_fini();
int send_steal_request(parsec_execution_stream_t* es);
int process_steal_request(parsec_execution_stream_t* es);
int parsec_node_mig_inc_task_executed();
int parsec_node_mig_get_task_executed();
int process_mig_request(parsec_task_t* this_task);
int process_mig_task_details(parsec_execution_stream_t* es);
int migrate_put_mpi_progress(parsec_execution_stream_t* es);
#endif