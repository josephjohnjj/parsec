#ifndef PARSEC_DEVICE_CUDA_MIGRATE_H
#define PARSEC_DEVICE_CUDA_MIGRATE_H

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/utils/zone_malloc.h"
#include "parsec/mca/device/cuda/device_cuda_internal.h"
#include "parsec/scheduling.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvml.h>

#define CUDA_DEVICE_NUM(DEVICE_NUM) (DEVICE_NUM - 2)
#define DEVICE_NUM(CUDA_DEVICE_NUM) (CUDA_DEVICE_NUM + 2)

#define TASK_NOT_MIGRATED 0
#define TASK_MIGRATED_BEFORE_STAGE_IN 1
#define TASK_MIGRATED_AFTER_STAGE_IN 2

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
    int level0;
    int level1;
    int level2;
    int received;
} parsec_device_cuda_info_t;

typedef struct migrated_task_s
{
    parsec_list_item_t list_item;
    parsec_gpu_task_t *gpu_task;
    parsec_device_gpu_module_t *dealer_device;
    parsec_device_gpu_module_t *starving_device;
    int stage_in_status;
} migrated_task_t;

typedef struct task_mapping_item_s
{
    parsec_hash_table_item_t ht_item;
    int device_index;
} task_mapping_item_t;

typedef struct gpu_dev_prof_s
{
    int32_t device_index;
    int32_t task_count;
} gpu_dev_prof_t;

int parsec_cuda_migrate_init(int ndevices);
int parsec_cuda_migrate_fini();
int parsec_cuda_get_device_load(int device);
int parsec_cuda_set_device_load(int device, int load);
int parsec_cuda_get_device_task(int device, int level);
int parsec_cuda_set_device_task(int device, int task_count, int level);
int parsec_cuda_tasks_executed(int device);
int is_starving(int device);
int find_starving_device(int dealer_device);
parsec_device_gpu_module_t *parsec_cuda_change_device(int dealer_device_index);
int parsec_cuda_mig_task_enqueue(parsec_execution_stream_t *es, migrated_task_t *mig_task);
int parsec_cuda_mig_task_dequeue(parsec_execution_stream_t *es);
int migrate_immediate(parsec_execution_stream_t *es, parsec_device_gpu_module_t *dealer_device,
                      parsec_gpu_task_t *migrated_gpu_task);
int migrate_to_starving_device(parsec_execution_stream_t *es, parsec_device_gpu_module_t *dealer_device);
int parsec_gpu_data_reserve_device_space_for_flow(parsec_device_gpu_module_t *gpu_device,
                                                  parsec_gpu_task_t *gpu_task, const parsec_flow_t *flow);
int migrate_data_d2d(parsec_gpu_task_t *gpu_task, parsec_device_gpu_module_t *src_dev,
                     parsec_device_gpu_module_t *dest_dev);
int change_task_features(parsec_gpu_task_t *gpu_task, parsec_device_gpu_module_t *dealer_device,
                         parsec_device_gpu_module_t *starving_device, int stage_in_status);
int gpu_data_version_increment(parsec_gpu_task_t *gpu_task, parsec_device_gpu_module_t *gpu_device);
int gurantee_ownership_transfer(parsec_gpu_task_t *gpu_task, parsec_data_t *data, int flow_index,
                                parsec_data_copy_t *src_copy, parsec_data_copy_t *dst_copy,
                                uint8_t stage_in_device, uint8_t access_mode);
double current_time();
int update_task_to_device_mapping(parsec_task_t *task, int device_index);
int find_task_to_device_mapping(parsec_task_t *task);
void clear_task_migrated_per_tp();
void print_task_migrated_per_tp();
int get_tp_count();

#endif