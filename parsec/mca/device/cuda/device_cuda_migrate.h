#ifndef PARSEC_DEVICE_CUDA_MIGRATE_H
#define PARSEC_DEVICE_CUDA_MIGRATE_H


#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/mca/device/cuda/device_cuda_internal.h"
#include "parsec/scheduling.h"
#include <cuda.h>
#include <cuda_runtime_api.h>


typedef struct parsec_device_cuda_info_s {
    int                       task_count;
    int                       load;
    //parsec_atomic_lock_t    lock;
} parsec_device_cuda_info_t;

int parsec_cuda_migrate_init(int ndevices);
int parsec_cuda_get_device_load(int device);
int parsec_cuda_get_device_task(int device);
int parsec_cuda_set_device_load(int device, int load);
int parsec_cuda_set_device_task(int device, int task_count);
int is_starving(int device);
int find_starving_device(int dealer_device);
parsec_device_gpu_module_t* parsec_cuda_change_device( int dealer_device_index);
int parsec_cuda_kernel_migrate( parsec_execution_stream_t *es,
                                int starving_device_index,
                                parsec_gpu_task_t *migrated_gpu_task);


#endif




