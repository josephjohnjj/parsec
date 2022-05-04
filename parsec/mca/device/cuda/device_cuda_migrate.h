#ifndef PARSEC_DEVICE_CUDA_MIGRATE_H
#define PARSEC_DEVICE_CUDA_MIGRATE_H


#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/mca/device/cuda/device_cuda_internal.h"
#include "parsec/scheduling.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvml.h>

#define CUDA_DEVICE_NUM(DEVICE_NUM) (DEVICE_NUM - 2)
#define DEVICE_NUM(CUDA_DEVICE_NUM) (CUDA_DEVICE_NUM + 2)

/**
 * @brief 
 * level 0 - task has been enqueued to the pending queue of the device. It has not been progressed.
 * level 1 - task has been dequeued from the pending queue of the device and it has been moved to 
 *           the queue that deals with movement of the task data to the GPU, but has not yet been moved
 * level 2 - task data has been moved to the GPU, GPU is in control of the data and Task.
 * 
 */
#define EXECUTION_LEVEL 3

typedef struct parsec_device_cuda_info_s 
{
    int total_tasks_executed;
    int task_count[EXECUTION_LEVEL];
    int load;
} parsec_device_cuda_info_t;

typedef struct migrated_task_s
{
    parsec_list_item_t list_item;
    parsec_gpu_task_t* gpu_task;
    parsec_device_gpu_module_t* dealer_device;
    parsec_device_gpu_module_t* starving_device;

} migrated_task_t;

int parsec_cuda_migrate_init(int ndevices);
int parsec_cuda_migrate_fini();
int parsec_cuda_get_device_load(int device);
int parsec_cuda_set_device_load(int device, int load);
int parsec_cuda_get_device_task(int device, int level);
int parsec_cuda_set_device_task(int device, int task_count, int level);
int parsec_cuda_tasks_executed(int device);
int is_starving(int device);
int find_starving_device(int dealer_device);
parsec_device_gpu_module_t* parsec_cuda_change_device( int dealer_device_index);
int parsec_cuda_mig_task_enqueue( parsec_execution_stream_t *es, migrated_task_t *mig_task);
int parsec_cuda_mig_task_dequeue( parsec_execution_stream_t *es);
int migrate_immediate(parsec_execution_stream_t *es,  parsec_device_gpu_module_t* dealer_device,
                      parsec_gpu_task_t* migrated_gpu_task);
int migrate_if_starving(parsec_execution_stream_t *es,  parsec_device_gpu_module_t* dealer_device);



#endif




