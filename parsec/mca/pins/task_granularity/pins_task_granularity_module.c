/*
 * Copyright (c) 2012-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <stdio.h>
#include <stdlib.h>

#include "parsec/parsec_config.h"
#include "pins_task_granularity.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/utils/debug.h"
#include "parsec/execution_stream.h"
#include "parsec/profiling.h"
#include "parsec/parsec_internal.h"
#include "parsec/os-spec-timing.h"

static void pins_init_task_granularity(parsec_context_t* master_context);
static void pins_fini_task_granularity(parsec_context_t* master_context);
static void pins_thread_init_task_granularity(parsec_execution_stream_t* es);
static void pins_thread_fini_task_granularity(parsec_execution_stream_t* es);

static FILE *file_ptr;
parsec_atomic_lock_t lock;

int task_granularity_trace_keyin;
int task_granularity_trace_keyout;

typedef struct task_characteristics_s
{
    int taskpool_id;
    int task_class_id;
    int nb_data_items;
    int total_data_size;
    int priority;
    int chore_id;
} task_characteristics_t;

const parsec_pins_module_t parsec_pins_task_granularity_module = {
    &parsec_pins_task_granularity_component,
    {
        pins_init_task_granularity,
        pins_fini_task_granularity,
        NULL,
        NULL,
        pins_thread_init_task_granularity,
        pins_thread_fini_task_granularity
    },
    { NULL }
};


static void start_task_granularity_record(parsec_execution_stream_t* es,
                                    parsec_task_t* task,
                                    parsec_pins_next_callback_t* data);

static void stop_task_granularity_record(parsec_execution_stream_t* es,
                                    parsec_task_t* task,
                                    parsec_pins_next_callback_t* data);


static void pins_init_task_granularity(parsec_context_t* master)
{
    (void)master;
    parsec_profiling_add_dictionary_keyword("TASK_GRANULARITY", "fill:#FF0000",
                                           sizeof(task_characteristics_t),
                                           "taskpool_id{int32_t};task_class_id{int32_t};nb_data_items{int32_t};total_data_size{int32_t};priority{int32_t};chore_id{int32_t}",
                                           &task_granularity_trace_keyin,
                                           &task_granularity_trace_keyout);

}

static void pins_fini_task_granularity(parsec_context_t* master)
{ 
    (void)master;
}

static void pins_thread_init_task_granularity(parsec_execution_stream_t* es)
{
    parsec_pins_next_callback_t* event_cb;

    event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
    PARSEC_PINS_REGISTER(es, EXEC_BEGIN, start_task_granularity_record, event_cb);
    event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
    PARSEC_PINS_REGISTER(es, EXEC_END, stop_task_granularity_record, event_cb);  
}

static void pins_thread_fini_task_granularity(parsec_execution_stream_t* es)
{
    task_characteristics_t characteristics;
    parsec_pins_next_callback_t* event_cb;

    PARSEC_PINS_UNREGISTER(es, EXEC_BEGIN, start_task_granularity_record, &event_cb);
    free(event_cb);
    PARSEC_PINS_UNREGISTER(es, EXEC_END, stop_task_granularity_record, &event_cb);
    free(event_cb); 
}

static void start_task_granularity_record(parsec_execution_stream_t* es,
                                    struct parsec_task_s* task,
                                    parsec_pins_next_callback_t* data)
{

    task_characteristics_t characteristics;

    PARSEC_PROFILING_TRACE(es->es_profile,
                           task_granularity_trace_keyin,
                           task->task_class->key_functions->key_hash(task->task_class->make_key(task->taskpool, task->locals), NULL),
                           task->task_class->key_functions->key_hash(task->task_class->make_key(task->taskpool, task->locals), NULL),
                           (void*)&characteristics);
}

int find_chore(parsec_execution_stream_t* es, parsec_task_t* task)
{
    const parsec_task_class_t* tc = task->task_class;
    uint8_t chore_mask = task->chore_mask;
    parsec_evaluate_function_t* eval;
    unsigned int chore_id;
    int rc;
    
    /* Find first bit in chore_mask that is not 0 */
    for(chore_id = 0; NULL != tc->incarnations[chore_id].hook; chore_id++)
        if( 0 != (chore_mask & (1<<chore_id)) )
            break;

    do {
        if( NULL != (eval = tc->incarnations[chore_id].evaluate) ) {
            rc = eval(task);
            if( PARSEC_HOOK_RETURN_DONE != rc ) {
                if( PARSEC_HOOK_RETURN_NEXT != rc ) {
                    break;
                }
                goto next_chore;
            }
        }

        return chore_id;

    next_chore:
        /* Mark this chore as tested */
        chore_mask &= ~( 1<<chore_id );
        /* Find next chore to try */
        for(chore_id = chore_id+1; NULL != tc->incarnations[chore_id].hook; chore_id++)
            if( 0 != (chore_mask & (1<<chore_id)) )
                break;
    } while(NULL != tc->incarnations[chore_id].hook);

    return PARSEC_HOOK_RETURN_ERROR;
}


int find_data_size(parsec_execution_stream_t* es, parsec_task_t* task)
{
    int i, total_data = 0, nb_elements = 0, size = 0;
    struct parsec_data_copy_s* task_data;

    for(i = 0; i < task->task_class->nb_flows; i++)
    {
        task_data = task->data[i].data_in;
        if(task_data == NULL)
            task_data = task->data[i].data_out;
               
        if(task_data != NULL)
        {   
            if(task_data->arena_chunk != NULL)
            {
                nb_elements = task_data->arena_chunk->count;
                if(task_data->arena_chunk->origin != NULL)
                    size = task_data->arena_chunk->origin->elem_size;
            }
            else if(task_data->original != NULL)
                total_data += task_data->original->nb_elts;
            else
                printf("SOMETHING IS WRONG Name %s Id %d \n", task->task_class->name, task->task_class->task_class_id);
             
        }
    }
    return total_data;
}



static void stop_task_granularity_record(parsec_execution_stream_t* es,
                                    parsec_task_t* task,
                                    parsec_pins_next_callback_t* data)
{
    task_characteristics_t characteristics;

    characteristics.taskpool_id = task->taskpool->taskpool_id;
    characteristics.task_class_id = task->task_class->task_class_id;
    characteristics.nb_data_items = task->task_class->nb_parameters;
    characteristics.total_data_size = find_data_size(es, task);
    characteristics.priority = task->priority;
    characteristics.chore_id = find_chore(es, task); 

    PARSEC_PROFILING_TRACE(es->es_profile,
                           task_granularity_trace_keyout,
                           task->task_class->key_functions->key_hash(task->task_class->make_key(task->taskpool, task->locals), NULL),
                           task->task_class->key_functions->key_hash(task->task_class->make_key(task->taskpool, task->locals), NULL),
                           (void*)&characteristics);

}


