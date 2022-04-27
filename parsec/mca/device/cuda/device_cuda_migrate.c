
#include "parsec/mca/device/cuda/device_cuda_migrate.h"

extern int parsec_device_cuda_enabled;
parsec_device_cuda_info_t* device_info; 
static parsec_list_t** migrated_task_list;
static int NDEVICES;


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
    #if defined(PARSEC_HAVE_CUDA)
    nvmlReturn_t nvml_ret;
    #endif

    NDEVICES = ndevices;
    device_info = (parsec_device_cuda_info_t *) calloc(ndevices, sizeof(parsec_device_cuda_info_t));
    migrated_task_list = (parsec_list_t**) calloc(ndevices, sizeof(parsec_list_t*));

    for(i = 0; i < NDEVICES; i++)
    {
        device_info[i].task_count = 0;
        device_info[i].load = 0;
        migrated_task_list[i] = PARSEC_OBJ_NEW(parsec_list_t);
    }

    #if defined(PARSEC_HAVE_CUDA)
    nvml_ret = nvmlInit_v2();
    #endif

    printf("Migration module initialised for %d devices \n", NDEVICES);

    return 0;

}

int parsec_cuda_migrate_fini()
{
    int i;


	
    #if defined(PARSEC_HAVE_CUDA)
    nvmlShutdown();
    #endif

    for(i = 0; i < NDEVICES; i++)
    {
        PARSEC_OBJ_RELEASE(migrated_task_list[i]); 
    }
    //free(migrated_task_list);

    //free(device_info); 


    printf("Migration module shut down \n");

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

    #if defined(PARSEC_HAVE_CUDA)
    nvmlDevice_t nvml_device;
    nvmlUtilization_t nvml_utilization;
    nvmlReturn_t nvml_ret;
    
    nvmlDeviceGetHandleByIndex_v2(device, &nvml_device);
    nvml_ret = nvmlDeviceGetUtilizationRates ( nvml_device, &nvml_utilization);
    device_info[device].load = nvml_utilization.gpu;

    printf("NVML Device Load GPU %d Memory %d \n", nvml_utilization.gpu, nvml_utilization.memory);
    #else
	device_info[device].load = device_info[device].task_count;
    #endif /* PARSEC_HAVE_CUDA */

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
    printf("Device %d: Current load %d, new load %d \n", 
        device, device_info[device].task_count, device_info[device].task_count+task_count);

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
int find_starving_device(int dealer_device)
{
    int i;

    // 0 device is the CPU, 1 is recursive
    for(i = 0; i < NDEVICES; i++)
    {
        if( i == dealer_device) 
            continue;

	//printf("Find_starving_device: Total_Dev %d  Dealer_Dev %d starving device %d\n", NDEVICES, dealer_device, i);
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

    starving_device_index = find_starving_device(dealer_device_index);
    
    if(starving_device_index == -1)
        starving_device_index = dealer_device_index;
    starving_gpu_device = (parsec_device_gpu_module_t*)parsec_mca_device_get(starving_device_index);

    printf(" parsec_cuda_change_device: Total_Dev %d  Dealer_Dev %d Starving_dev %d \n", 
    parsec_device_cuda_enabled, dealer_device_index, starving_device_index);

    return starving_gpu_device;
}



int parsec_cuda_kernel_enqueue( parsec_execution_stream_t *es,
                        parsec_task_t            *task,
                        int   starving_device_index)
{
    parsec_list_t* li = migrated_task_list[starving_device_index];
    parsec_list_chain_sorted(li, (parsec_list_item_t*) task, parsec_execution_context_priority_comparator);

    printf("Migrated task enqueued to the recieved_task_queue  of device %d \n", starving_device_index);

    return 0;
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

int parsec_cuda_kernel_dequeue( parsec_execution_stream_t *es)
{
    int i;
    parsec_task_t * task = NULL;
    parsec_list_t* li = NULL;

    for(i = 0; i < NDEVICES; i++)
    {
        li = migrated_task_list[i];
        task = (parsec_task_t*) parsec_list_pop_front(li);
        if(task != NULL)
            break;
    }

    if(task != NULL)  
    { 

        printf("Migrated task dequeued from the recieved_task_queue of device %d and scheduled to the device %d\n", i, i);
        parsec_cuda_kernel_scheduler(es, task, i+2); /* device 0 is the CPU, device 1 is recursive, cuda device count starts from 0 */ 
    }
}

/**
 * This function migrate a specific task from a device a
 * to another. 
 *
 *  Returns: negative number if any error occured.
 *           positive: starving device index.
 */
int parsec_cuda_kernel_migrate( parsec_execution_stream_t *es,
                                int starving_device_index,
                                parsec_gpu_task_t *migrated_gpu_task)
{
    parsec_cuda_kernel_enqueue(es, (parsec_task_t *) migrated_gpu_task, starving_device_index); 
    parsec_cuda_set_device_task(starving_device_index, 1);

    return starving_device_index;
}

/**
 * @brief check if there are any devices starving. If there are any starving device migrate
 * half the task from the dealer device to the starving device.
 * 
 * @param es 
 * @param dealer_gpu_device 
 * @return int 
 */

int migrate_if_starving(parsec_execution_stream_t *es,  parsec_device_gpu_module_t* dealer_device)
{
    int starving_device_index = -1, dealer_device_index = 0, dealer_task_count = 0;
    int half = 0, nb_migrated = 0;
    parsec_gpu_task_t *migrated_gpu_task = NULL;

    if(dealer_device->mutex < 3) // make sure dealer does not starve
        return -1;
    
    //dealer_task_count = parsec_cuda_get_device_task(dealer_device_index);
    dealer_device_index = dealer_device->super.device_index;
    starving_device_index = find_starving_device(dealer_device_index);
    if(starving_device_index == -1)
        return -1;

    //do
    //{
        migrated_gpu_task = (parsec_gpu_task_t*)parsec_fifo_try_pop( &(dealer_device->pending) );
        if(migrated_gpu_task != NULL)
        {
            nb_migrated++;
            parsec_cuda_set_device_load(dealer_device_index, -1); // decrement task count at the dealer device
            parsec_cuda_set_device_load(starving_device_index, 1); // increment task count at the starving device
            parsec_cuda_kernel_migrate(es, starving_device_index, migrated_gpu_task);
            printf("Tasks migrated from device %d to device %d: %d \n", dealer_device_index, starving_device_index, nb_migrated);
        }
        //else
        //    break;

        //half++;
    //}while(half < (dealer_task_count / 2) );

    //if(nb_migrated > 0)
    //	printf("Tasks migrated from device %d to device %d: %d \n", dealer_device_index, starving_device_index, nb_migrated);

    return nb_migrated;
}



/**
 * @brief Tasks is migrated immediatly. 
 * Mainly used for validating migration protocol.
 * 
 * @param es 
 * @param dealer_device 
 * @param migrated_gpu_task 
 * @return int 
 */

int migrate_immediate(parsec_execution_stream_t *es,  parsec_device_gpu_module_t* dealer_device,
                      parsec_gpu_task_t* migrated_gpu_task)
{
    int starving_device_index = -1, dealer_device_index = 0, dealer_task_count = 0;
    int half = 0;

    starving_device_index = find_starving_device(dealer_device_index);
    if(starving_device_index == -1)
        return -1;

    dealer_device_index = dealer_device->super.device_index;

    
    if(migrated_gpu_task != NULL)
    {
        parsec_cuda_set_device_load(dealer_device_index, -1); // decrement task count at the dealer device
        parsec_cuda_set_device_load(starving_device_index, 1); // increment task count at the starving device
        parsec_cuda_kernel_migrate(es, starving_device_index, migrated_gpu_task);
        return 1;
    }
    
    return 0;

       
}


