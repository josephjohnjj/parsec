
#include "parsec/mca/device/cuda/device_cuda_migrate.h"

extern int parsec_device_cuda_enabled;
static parsec_device_cuda_info_t* device_info; 


/**
 * @brief The function initialises the data structures required
 * for inter-device migration.
 * 
 * @param ndevices number of devices
 * @return int 
 */

int parsec_cuda_migrate_init(int ndevices)
{
    int i;
    cudaError_t cudastatus;
    nvmlReturn_t nvml_ret;

    device_info = (parsec_device_cuda_info_t *) calloc(ndevices, sizeof(parsec_device_cuda_info_t));

    for(i = 0; i < ndevices; i++)
    {
        device_info[i].task_count = 0;
        device_info[i].load = 0;
    }

    nvml_ret = nvmlInit_v2();

    return 0;

}

int parsec_cuda_migrate_fini(int ndevices)
{
    free(device_info); 
    nvmlShutdown();

    return 0;

}

/**
 * @brief returns the load of a particular device
 * 
 * nvml_utilization has two fields - gpu and memory
 * gpu - Percent of time over the past sample period during which one or more kernels was executing on the GPU.
 * memory - Percent of time over the past sample period during which global (device) memory was being read or written
 * 
 * @param device index of the device
 * @return int 
 */

int parsec_cuda_get_device_load(int device)
{
    unsigned int nvml_dev;
    nvmlDevice_t nvml_device;
    nvmlUtilization_t nvml_utilization;
    
    nvmlDeviceGetHandleByIndex_v2(device, &nvml_device);
    nvml_ret = nvmlDeviceGetUtilizationRates ( nvml_device, &nvml_utilization);
    device_info[device].load = nvml_utilization.gpu;

    printf("NVML Device Load GPU %d Memory %d \n", nvml_utilization.gpu, nvml_utilization.memory);

    return device_info[device].load;
 
}

/**
 * @brief returns the number of tasks in a particular device
 * 
 * @param device index of the device
 * @return int 
 */

int parsec_cuda_get_device_task(int device)
{
    return device_info[device].task_count;
}


/**
 * @brief sets the load of a particular device
 * 
 * @param device index of the device
 * @return int 
 */

int parsec_cuda_set_device_load(int device, int load)
{
    device_info[device].load += load;
    return device_info[device].load;
}

/**
 * @brief sets the number of tasks in a particular device
 * 
 * @param device index of the device
 * @return int 
 */

int parsec_cuda_set_device_task(int device, int task_count)
{
    device_info[device].task_count += task_count;
    return device_info[device].task_count;
}

/**
 * @brief returns 1 if the device is starving, 0 if its is not
 * 
 * @param device index number of the device
 * @return int 
 *
 * TODO: needs updation
 */
int is_starving(int device)
{
    if( device_info[device].load < 1 && device_info[device].task_count < 1 )
        return 1;
    else
        return 0;
}

/**
 * @brief returns the index of a starving device and returns -1
 * if no device is starving.
 * 
 * @param dealer_device device probing for a starving device
 * @param ndevice total number of devices
 * @return int 
 * 
 * TODO: needs updation
 */
int find_starving_device(int dealer_device, int ndevice)
{
    int i;
    printf(" find_starving_device: Total_Dev %d  Dealer_Dev %d\n", ndevice, dealer_device);

    // 0 device is the CPU, 1 is recursive
    for(i = 2; i < (2 + ndevice); i++)
    {
        printf("Trying_Dev %d  Dealer_Dev %d\n", i, dealer_device);
        if( i == dealer_device) 
            continue;

        //if(is_starving(i))
            return i;
    }

    return -1; 
}


/**
 * @brief selects a new starving device instead of the originally
 * intended device. This enables migration of a task before it
 * is scheduled to any particular device.
 * 
 * @param dealer_device_index the device the task was initially assigned to
 * @return parsec_device_gpu_module_t* 
 * 
 */
parsec_device_gpu_module_t*
parsec_cuda_change_device( int dealer_device_index)
{
    int starving_device_index;
    parsec_device_gpu_module_t* starving_gpu_device;

    printf("parsec_cuda_change_device: Total_Dev %d  Dealer_Dev %d\n", 
        parsec_device_cuda_enabled, dealer_device_index);
    
    starving_device_index = find_starving_device(dealer_device_index, parsec_device_cuda_enabled);
    
    if(starving_device_index == -1)
        starving_device_index = dealer_device_index;
    starving_gpu_device = (parsec_device_gpu_module_t*)parsec_mca_device_get(starving_device_index);

    printf(" Starving_dev %d \n", starving_device_index);

    return starving_gpu_device;
}


/**
 * This function migrate a specific task from a device a
 * to another. 
 *
 *  Returns: negative number if any error occured.
 *           positive: starving device index.
 */
int parsec_cuda_kernel_migrate( parsec_execution_stream_t *es,
                        parsec_device_gpu_module_t   *dealer_device,
                        parsec_gpu_task_t            *migrated_gpu_task)
{
    printf("TRIAL parsec_cuda_kernel_migrate \n");

    int starving_device_index, dealer_device_index;
    parsec_device_gpu_module_t* starving_gpu_device;
    
    dealer_device_index = dealer_device->super.device_index;
    starving_device_index = find_starving_device(dealer_device_index, parsec_device_cuda_enabled);

    if(starving_device_index == -1)
        return -1;

    /**
     * @brief The distance value in normal parsec scheduler is laways positive. So a negative
     * distance value can be used to communicate the device index of the staving node to the 
     * qpd scheduler. The distance is calucaled as distance = ( (starving device index) * -1 ) -1
     * 
     */
    __parsec_schedule(es, (parsec_task_t *) migrated_gpu_task, (starving_device_index * -1) - 1); 
    printf("Task migrated to device %d \n", starving_device_index);

    return starving_device_index;
}

