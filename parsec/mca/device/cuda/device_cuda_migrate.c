#include "parsec/mca/device/cuda/device_cuda_migrate.h"

extern int parsec_device_cuda_enabled;
parsec_device_cuda_info_t* device_info; 
static parsec_list_t* migrated_task_list;
static int NDEVICES;
migration_accounting_t* accounting;


/**
 * @brief The function initialises the data structures required
 * for inter-device migration.
 * 
 * @param ndevices number of devices
 * @return int 
 */

int parsec_cuda_migrate_init(int ndevices)
{
    int i, j;
    
    #if defined(PARSEC_HAVE_CUDA)
    nvmlReturn_t nvml_ret;
    #endif

    NDEVICES = ndevices;
    device_info = (parsec_device_cuda_info_t *) calloc(ndevices, sizeof(parsec_device_cuda_info_t));
    accounting = (migration_accounting_t *) calloc(ndevices, sizeof(migration_accounting_t));
    migrated_task_list = PARSEC_OBJ_NEW(parsec_list_t);;

    for(i = 0; i < NDEVICES; i++)
    {
        for(j = 0; j < EXECUTION_LEVEL; j++)
            device_info[i].task_count[j] = 0;
        device_info[i].load = 0;
        
        accounting[i].level0 = 0;
        accounting[i].level1 = 0;
        accounting[i].level2 = 0;
        accounting[i].total_tasks_executed = 0;
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
    int total_tasks = 0;
	
    #if defined(PARSEC_HAVE_CUDA)
    nvmlShutdown();
    #endif

    for(i = 0; i < NDEVICES; i++)
    {
        printf("*********** DEVICE %d *********** \n", i);
        printf("Total tasks executed: %d \n", accounting[i].total_tasks_executed);
            printf("Tasks migrated: level0 %d, level1 %d, level2 %d (Total %d)\n",
            accounting[i].level0, accounting[i].level1, accounting[i].level2,
            accounting[i].level0 + accounting[i].level1 + accounting[i].level2);
    }
    PARSEC_OBJ_RELEASE(migrated_task_list); 
    free(device_info); 

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
 * @brief sets the load of a particular device
 * 
 * @param device index of the device
 * @return int 
 */

int parsec_cuda_set_device_load(int device, int load)
{
    int rc = parsec_atomic_fetch_add_int32(&(device_info[device].load), load);
    return rc + load;
}


/**
 * @brief returns the number of tasks in a particular device
 * 
 * @param device index of the device
 * @param level level of execution
 * @return int 
 */

int parsec_cuda_get_device_task(int device, int level)
{
    return device_info[device].task_count[level];
}


/**
 * @brief sets the number of tasks in a particular device
 * 
 * @param device index of the device
 * @param level level of execution
 * @return int 
 */

int parsec_cuda_set_device_task(int device, int task_count, int level)
{
    int rc = parsec_atomic_fetch_add_int32(&(device_info[device].task_count[level]), task_count);
    return rc + task_count;
}


/**
 * @brief sets the load of a particular device
 * 
 * @param device index of the device
 * @return int 
 */

int parsec_cuda_tasks_executed(int device)
{
    int rc = parsec_atomic_fetch_add_int32(&(accounting[device].total_tasks_executed), 1);
    return rc + 1;
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
    //if( device_info[device].load < 1 && device_info[device].task_count < 1 )
    if( device_info[device].task_count[/* level */ 0] < 1 )
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

    for(i = 0; i < NDEVICES; i++)
    {
        if( i == dealer_device ) 
            continue;

        if(is_starving(i))
            return i;
    }

    return -1; 
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

int parsec_cuda_mig_task_dequeue( parsec_execution_stream_t *es)
{
    char tmp[128];
    migrated_task_t *mig_task = NULL;
    parsec_gpu_task_t *migrated_gpu_task;
    parsec_device_gpu_module_t* dealer_device;
    parsec_device_gpu_module_t* starving_device;

    mig_task = (migrated_task_t*) parsec_fifo_try_pop(migrated_task_list);

    if(mig_task != NULL)  
    { 
        parsec_gpu_task_t *migrated_gpu_task = mig_task->gpu_task;
        parsec_device_gpu_module_t* dealer_device = mig_task->dealer_device;
        parsec_device_gpu_module_t* starving_device = mig_task->starving_device;
        change_task_features(migrated_gpu_task, dealer_device);

	    PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t*)migrated_gpu_task);
        printf("Dequeue task %s from device queue and schedule\n", parsec_task_snprintf(tmp, MAX_TASK_STRLEN, ((parsec_gpu_task_t *) migrated_gpu_task)->ec));	
        parsec_cuda_kernel_scheduler(es, (parsec_gpu_task_t *) migrated_gpu_task, starving_device->super.device_index);  
        free(mig_task);

        return 1;
    }

    return 0;
}

/**
 * This function migrate a specific task from a device a
 * to another. 
 *
 *  Returns: negative number if any error occured.
 *           positive: starving device index.
 */
int parsec_cuda_mig_task_enqueue( parsec_execution_stream_t *es, migrated_task_t *mig_task)
{
    //parsec_list_chain_sorted(migrated_task_list, (parsec_list_item_t*) mig_task, parsec_execution_context_priority_comparator);
     parsec_list_push_back((parsec_list_t*)migrated_task_list, (parsec_list_item_t*)mig_task);
    
    char tmp[MAX_TASK_STRLEN];
    parsec_gpu_task_t *migrated_gpu_task = mig_task->gpu_task;
    parsec_device_gpu_module_t* dealer_device = mig_task->dealer_device;
    parsec_device_gpu_module_t* starving_device = mig_task->starving_device;
    printf("Enqueue task %s to device queue %d\n", parsec_task_snprintf(tmp, MAX_TASK_STRLEN, 
        ((parsec_gpu_task_t *) migrated_gpu_task)->ec), CUDA_DEVICE_NUM(starving_device->super.device_index));

    return 0;
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
    parsec_device_gpu_module_t* starving_device = NULL;
    migrated_task_t *mig_task = NULL;
    char tmp[128];

    dealer_device_index = CUDA_DEVICE_NUM(dealer_device->super.device_index);  
    if(is_starving(dealer_device_index))
        return 0;
    
    starving_device_index = find_starving_device(dealer_device_index);
    if(starving_device_index == -1)
        return 0;
    starving_device = (parsec_device_gpu_module_t*)parsec_mca_device_get(DEVICE_NUM(starving_device_index));

    do
    {
        migrated_gpu_task = (parsec_gpu_task_t*)parsec_fifo_try_pop( &(dealer_device->pending) );
        if(migrated_gpu_task != NULL)
        {
            /**
             * @brief if the GPU task is a not a computational kerenel
             * stop migration.
             */
            if(migrated_gpu_task->task_type != PARSEC_GPU_TASK_TYPE_KERNEL)
            {
                parsec_list_push_front(&(dealer_device->pending), (parsec_list_item_t*) migrated_gpu_task);
                return nb_migrated;
            }
	    
	        PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t*)migrated_gpu_task);
            nb_migrated++;
            parsec_cuda_set_device_task(dealer_device_index, /* count */ -1, /* level */ 0); // decrement task count at the dealer device
	        printf("Task %s migrated from device %d to device %d: nb_migrated %d\n", parsec_task_snprintf(tmp, MAX_TASK_STRLEN, ((parsec_gpu_task_t *) migrated_gpu_task)->ec), dealer_device_index, starving_device_index, nb_migrated);

            //change_task_features(migrated_gpu_task, dealer_device);
            mig_task = (migrated_task_t *) malloc(sizeof(migrated_task_t));
            mig_task->gpu_task = migrated_gpu_task;
            mig_task->dealer_device = dealer_device;
            mig_task->starving_device = starving_device;

	        parsec_cuda_mig_task_enqueue(es, mig_task);
        }
        else
            break;

        half++;
    }while(half < (dealer_task_count / 2) );

    return nb_migrated;
}

int change_task_features(parsec_gpu_task_t *gpu_task, parsec_device_gpu_module_t* dealer_device)
{
    int i = 0;
    parsec_task_t *task = gpu_task->ec;
    char tmp[128];

    for(i = 0; i < task->task_class->nb_flows; i++)
    {
        if (task->data[i].data_out == NULL)
            continue;

        if(task->data[i].data_out->original->owner_device == dealer_device->super.device_index)
        {
            if(PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & gpu_task->flow[i]->flow_flags)) //CTL flow
                continue;
            if(PARSEC_FLOW_ACCESS_WRITE  & gpu_task->flow[i]->flow_flags)
            {
                printf("%s: has WRITE permission on copy %p [reader = %d], in GPU[%s] with Coherency %d \n",
                    parsec_task_snprintf(tmp, MAX_TASK_STRLEN, ((parsec_gpu_task_t *)gpu_task)->ec),
                    task->data[i].data_out, task->data[i].data_out->readers,
                    dealer_device->super.name, task->data[i].data_out->coherency_state);
                printf("NEW permissions %d OLD permissions %d\n", gpu_task->flow[i]->flow_flags,
                    gpu_task->flow[i]->flow_flags ^ PARSEC_FLOW_ACCESS_WRITE);
            }

            if(PARSEC_FLOW_ACCESS_READ & gpu_task->flow[i]->flow_flags)
                printf("%s: has READ permission on copy %p [reader = %d], in GPU[%s] with Coherency %d \n",
                    parsec_task_snprintf(tmp, MAX_TASK_STRLEN, ((parsec_gpu_task_t *)gpu_task)->ec),
                    task->data[i].data_out, task->data[i].data_out->readers,
                    dealer_device->super.name, task->data[i].data_out->coherency_state);

            //parsec_device_cuda_module_t *in_elem_dev = (parsec_device_cuda_module_t*)parsec_mca_device_get( task->data[i].data_out->device_index );
            //printf("%s: %s (%s) can directly do D2D from devices with masks %d \n", 
            //    parsec_task_snprintf(tmp, MAX_TASK_STRLEN, ((parsec_gpu_task_t *)gpu_task)->ec), 
            //    dealer_device->super.name,(&in_elem_dev->super)->super.name,
            //    dealer_device->peer_access_mask);
            
            printf("%s: possible D2D from cuda device %d for flow %d  (from copy %p) \n", 
                parsec_task_snprintf(tmp, MAX_TASK_STRLEN, ((parsec_gpu_task_t *)gpu_task)->ec), 
                ((parsec_device_cuda_module_t* )dealer_device)->cuda_index,
                gpu_task->flow[i]->flow_index, task->data[i].data_out);

            task->data[i].data_in = task->data[i].data_out;
            if(task->data[i].data_in->device_index != task->data[i].data_in->original->owner_device)
            {
                printf("There is something wrong!! device index is %d instead of %d \n",
                task->data[i].data_in->device_index,
                task->data[i].data_in->original->owner_device);

                task->data[i].data_in->device_index = dealer_device->super.device_index;
            }
            parsec_data_t* original = task->data[i].data_in->original;
            parsec_list_item_ring_chop((parsec_list_item_t*)task->data[i].data_in);
            PARSEC_LIST_ITEM_SINGLETON(task->data[i].data_in);
            parsec_atomic_lock( &original->lock );
            if(task->data[i].data_in->readers < 0)
            {
                printf("There is something wrong!! reader is negative \n");
                task->data[i].data_in->readers = 0;
            }
            parsec_atomic_fetch_inc_int32(&task->data[i].data_in->readers);
            task->data[i].data_in->coherency_state = PARSEC_DATA_COHERENCY_SHARED;
            task->data[i].data_in->data_transfer_status = PARSEC_DATA_STATUS_SHOULD_MIGRATE;
            parsec_atomic_unlock( &original->lock );
        }
    }

    return 0;
}
