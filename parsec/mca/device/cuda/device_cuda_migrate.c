#include "parsec/mca/device/cuda/device_cuda_migrate.h"

extern int parsec_device_cuda_enabled;
parsec_device_cuda_info_t* device_info; 
static parsec_list_t* migrated_task_list;
static int NDEVICES;

double start = 0;
double end = 0;

PARSEC_OBJ_CLASS_INSTANCE(migrated_task_t, parsec_list_item_t, NULL, NULL);

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

    start = MPI_Wtime();

    NDEVICES = ndevices;
    device_info = (parsec_device_cuda_info_t *) calloc(ndevices, sizeof(parsec_device_cuda_info_t));
    migrated_task_list = PARSEC_OBJ_NEW(parsec_list_t);;

    for(i = 0; i < NDEVICES; i++)
    {
        for(j = 0; j < EXECUTION_LEVEL; j++)
            device_info[i].task_count[j] = 0;
        device_info[i].load = 0;

        device_info[i].level0 = 0;
        device_info[i].level1 = 0;
        device_info[i].level2 = 0;
        device_info[i].total_tasks_executed = 0;
        device_info[i].received = 0;
    }

    #if defined(PARSEC_HAVE_CUDA)
    nvml_ret = nvmlInit_v2();
    #endif

    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    //sleep(60);

    printf("Migration module initialised for %d devices \n", NDEVICES);


    return 0;

}

int parsec_cuda_migrate_fini()
{
    int i;

    end = MPI_Wtime();
	
    #if defined(PARSEC_HAVE_CUDA)
    nvmlShutdown();
    #endif

    for(i = 0; i < NDEVICES; i++)
    {
        printf("\n*********** DEVICE %d *********** \n", i);
        printf("Total tasks executed: %d \n", device_info[i].total_tasks_executed);
        printf("Tasks migrated      : level0 %d, level1 %d, level2 %d (Total %d)\n",
            device_info[i].level0, device_info[i].level1, device_info[i].level2,
            device_info[i].level0 + device_info[i].level1 + device_info[i].level2);
        printf("Task check          : level0 %d level1 %d level2 %d total %d \n", 
            parsec_cuda_get_device_task(i, 0),
            parsec_cuda_get_device_task(i, 1), 
            parsec_cuda_get_device_task(i, 2),
            parsec_cuda_get_device_task(i, -1));
        printf("Task received       : %d \n", device_info[i].received);
        
    }
    printf("\n---------Execution time = %lf ------------ \n", end - start); 
    PARSEC_OBJ_RELEASE(migrated_task_list); 
    free(device_info); 

    printf("Migration module shut down \n");

    return 0;

}


double current_time()
{
    return ( MPI_Wtime() - start);
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
    #if defined(PARSEC_HAVE_CUDA)
    nvmlDevice_t nvml_device;
    nvmlUtilization_t nvml_utilization;
    nvmlReturn_t nvml_ret;
    
    nvmlDeviceGetHandleByIndex_v2(device, &nvml_device);
    nvml_ret = nvmlDeviceGetUtilizationRates ( nvml_device, &nvml_utilization);
    device_info[device].load = nvml_utilization.gpu;

    //printf("NVML Device Load GPU %d Memory %d \n", nvml_utilization.gpu, nvml_utilization.memory);
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
    if( level == -1)
        return (device_info[device].task_count[0] +
                device_info[device].task_count[1] +
                device_info[device].task_count[2]);
                
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
    int rc = parsec_atomic_fetch_add_int32(&(device_info[device].total_tasks_executed), 1);
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
    return ( parsec_cuda_get_device_task(device, -1) < 1 ) ? 1 : 0;

}


int will_starve(int device)
{
    return ( parsec_cuda_get_device_task(device, -1) <  3 ) ? 1 : 0;
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
    parsec_gpu_task_t *migrated_gpu_task = NULL;
    parsec_device_gpu_module_t* dealer_device = NULL;
    parsec_device_gpu_module_t* starving_device = NULL;
    int stage_in_status = 0;


    mig_task = (migrated_task_t*) parsec_list_try_pop_front(migrated_task_list);

    if(mig_task != NULL)  
    { 
        PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t*)mig_task);
        migrated_gpu_task = mig_task->gpu_task;
        assert( migrated_gpu_task->migrate_status != TASK_NOT_MIGRATED );
        dealer_device = mig_task->dealer_device;
        starving_device = mig_task->starving_device;
        stage_in_status = mig_task->stage_in_status;

        change_task_features(migrated_gpu_task, dealer_device, stage_in_status);

	    PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t*)migrated_gpu_task);	
        parsec_atomic_fetch_inc_int32(&device_info[CUDA_DEVICE_NUM(starving_device->super.device_index)].received); 
        parsec_cuda_kernel_scheduler(es, (parsec_gpu_task_t *) migrated_gpu_task, starving_device->super.device_index);  
	    PARSEC_OBJ_DESTRUCT(mig_task);
        free(mig_task);

        return 1;
    }

    return 0;
}



/**
 * This function enqueues the migrated task to a node level queue.
 *
 *  Returns: negative number if any error occured.
 *           positive: starving device index.
 */
int parsec_cuda_mig_task_enqueue( parsec_execution_stream_t *es, migrated_task_t *mig_task)
{
    parsec_list_push_back((parsec_list_t*)migrated_task_list, (parsec_list_item_t*)mig_task);
    
    parsec_gpu_task_t *migrated_gpu_task = mig_task->gpu_task;
    parsec_device_gpu_module_t* starving_device = mig_task->starving_device;
    char tmp[MAX_TASK_STRLEN];
    PARSEC_DEBUG_VERBOSE(10, "Enqueue task %s to device queue %d", parsec_task_snprintf(tmp, MAX_TASK_STRLEN, 
        ((parsec_gpu_task_t *) migrated_gpu_task)->ec), CUDA_DEVICE_NUM(starving_device->super.device_index));

    (void)es;
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

int migrate_to_starving_device(parsec_execution_stream_t *es,  parsec_device_gpu_module_t* dealer_device)
{
    int starving_device_index = -1, dealer_device_index = 0;
    int nb_migrated = 0, execution_level = 0, stream_index = 0, j = 0;
    parsec_gpu_task_t *migrated_gpu_task = NULL;
    parsec_device_gpu_module_t* starving_device = NULL;
    migrated_task_t *mig_task = NULL;

    dealer_device_index = CUDA_DEVICE_NUM(dealer_device->super.device_index);  
    if( will_starve(dealer_device_index) )
        return 0;
    
    starving_device_index = find_starving_device(dealer_device_index);
    if(starving_device_index == -1)
        return 0;
    starving_device = (parsec_device_gpu_module_t*)parsec_mca_device_get(DEVICE_NUM(starving_device_index));

    /**
     * @brief Tasks are searched in different levels one by one. At this point we assume
     * that the cost of migration increases, as the level increase.
     */
    migrated_gpu_task = (parsec_gpu_task_t*)parsec_list_try_pop_back( &(dealer_device->pending) ); //level 0
    execution_level = 0;
    if(migrated_gpu_task == NULL)
    {
        migrated_gpu_task = (parsec_gpu_task_t*)parsec_list_try_pop_back( dealer_device->exec_stream[0]->fifo_pending ); //level 1
        execution_level = 1;

        if( migrated_gpu_task == NULL)
        {
            for(j = 0; j < (dealer_device->max_exec_streams - 2); j++)
            {
                migrated_gpu_task = (parsec_gpu_task_t*)parsec_list_try_pop_back( dealer_device->exec_stream[ (2 + j) ]->fifo_pending ); //level2
                if(migrated_gpu_task != NULL)
                {
                    execution_level = 2;
                    stream_index = 2 + j;
                    break;
                }
            }
        }
    }
    

    if(migrated_gpu_task != NULL)
    {
	    assert(migrated_gpu_task->ec != NULL);
        parsec_list_item_ring_chop( (parsec_list_item_t*)migrated_gpu_task );
        PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t*)migrated_gpu_task);
        /**
         * @brief if the task is a not a computational kerenel or if it is a task that has
         * already been migrated, we stop the migration.
         */
        if(migrated_gpu_task->task_type != PARSEC_GPU_TASK_TYPE_KERNEL || migrated_gpu_task->migrate_status > TASK_NOT_MIGRATED)
        {
            if(execution_level == 0)
                parsec_list_push_back(&(dealer_device->pending), (parsec_list_item_t*) migrated_gpu_task );
            if(execution_level == 1)
                parsec_list_push_back( dealer_device->exec_stream[0]->fifo_pending, (parsec_list_item_t*) migrated_gpu_task );
            if(execution_level == 2)
                parsec_list_push_back( dealer_device->exec_stream[stream_index]->fifo_pending, (parsec_list_item_t*) migrated_gpu_task );
                
            return nb_migrated;
        }

        assert( (migrated_gpu_task != NULL) && (migrated_gpu_task->ec != NULL) );

        if(execution_level == 0)
        {
            parsec_cuda_set_device_task(dealer_device_index, /* count */ -1, /* level */ 0); 
            device_info[dealer_device_index].level0++;
        }
        if(execution_level == 1)
        {
            parsec_cuda_set_device_task(dealer_device_index, /* count */ -1, /* level */ 1); 
            device_info[dealer_device_index].level1++;
        }
        if(execution_level == 2)
        {
            parsec_cuda_set_device_task(dealer_device_index, /* count */ -1, /* level */ 2); 
            device_info[dealer_device_index].level2++;
        }
        nb_migrated++;

        /**
         * @brief change migrate_status according to the status of the stage in of the
         * stage_in data.
         */
        if( execution_level == 2 )
            migrated_gpu_task->migrate_status = TASK_MIGRATED_AFTER_STAGE_IN; 
        else
            migrated_gpu_task->migrate_status = TASK_MIGRATED_BEFORE_STAGE_IN;

        /**
         * @brief An object of type migrated_task_t is created store the migrated task
         * and other associated details. This object is enqueued to a node level queue.
         * The main objective of this was to make sure that the manager does not have to sepend 
         * time on migration. It can select the task for migration, enqqueue it to the node level
         * queue and then return to its normal working. 
         */
        mig_task = (migrated_task_t *) calloc(1, sizeof(migrated_task_t));
	    PARSEC_OBJ_CONSTRUCT(mig_task, parsec_list_item_t);
        mig_task->gpu_task = migrated_gpu_task;
        mig_task->dealer_device = dealer_device;
        mig_task->starving_device = starving_device;
        mig_task->stage_in_status = (execution_level == 2) ? TASK_MIGRATED_AFTER_STAGE_IN : TASK_MIGRATED_BEFORE_STAGE_IN;
	    PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t*)mig_task);

	    parsec_cuda_mig_task_enqueue(es, mig_task);

        char tmp[MAX_TASK_STRLEN];
        PARSEC_DEBUG_VERBOSE(10, "Task %s migrated (level %d, stage_in %d) from device %d to device %d: nb_migrated %d", 
            parsec_task_snprintf(tmp, MAX_TASK_STRLEN, ((parsec_gpu_task_t *) migrated_gpu_task)->ec), 
            execution_level, mig_task->stage_in_status, dealer_device_index, starving_device_index, nb_migrated);
    }
    
    migrated_gpu_task = NULL;
    ///* update the expected load on the GPU device */
    parsec_device_load[dealer_device->super.device_index] -= nb_migrated * parsec_device_sweight[dealer_device->super.device_index];
    return nb_migrated;
}

/**
 * @brief This function changes the features of a task, in a way that it is preped
 * for migration.
 * 
 * @param gpu_task 
 * @param dealer_device 
 * @param stage_in_status 
 * @return int 
 */

int change_task_features(parsec_gpu_task_t *gpu_task, parsec_device_gpu_module_t* dealer_device,
                         int stage_in_status)
{
    int i = 0;
    parsec_task_t *task = gpu_task->ec;
    parsec_data_copy_t *src_copy = NULL;

    for(i = 0; i < task->task_class->nb_flows; i++)
    {
        if (task->data[i].data_out == NULL)
            continue;
        if(PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & gpu_task->flow[i]->flow_flags)) //CTL flow
            continue;

        /**
         * Data is already staged in the dealer device and we can find all the data
         * of the tasks to be migrated in the dealer device.
         */
        if( stage_in_status == TASK_MIGRATED_AFTER_STAGE_IN ) 
        {   
            parsec_data_t* original = task->data[i].data_out->original;
            parsec_atomic_lock( &original->lock );

            /**
             * @brief If the task is stage in the data is already available in the 
             * dealer GPU. So we can set data_in = data_out, in order to make sure 
             * that the source data for the second stage in is always selected as the
             * data in the delaer GPU.
             */
            task->data[i].data_in = task->data[i].data_out;
            task->data[i].data_in->coherency_state = PARSEC_DATA_COHERENCY_SHARED;
            PARSEC_OBJ_RETAIN(task->data[i].data_in);

            /**
             * @brief If the task only WRITE access, then we have to increment the
             * reader of the data_in, so that it does not go into negative value
             * when we call complete_stage( parsec_gpu_callback_complete_push() ) 
             * after the second stage in of the task is completed (on the starving device).
             * 
             * If the task only as READ access it is already in the gpu_mem_owned_lru of
             * the dealer device. If it has WRITE and READ-WRITE access we move the data
             * to gpu_mem_owned_lru.
             */
            if( (PARSEC_FLOW_ACCESS_READ & gpu_task->flow[i]->flow_flags) &&
                !(PARSEC_FLOW_ACCESS_WRITE & gpu_task->flow[i]->flow_flags)) 
            {
                assert(  task->data[i].data_in->readers > 0 );
            }
            if( !(PARSEC_FLOW_ACCESS_READ & gpu_task->flow[i]->flow_flags) &&
                (PARSEC_FLOW_ACCESS_WRITE & gpu_task->flow[i]->flow_flags)) 
            {
                assert(task->data[i].data_in->readers >= 0);
                PARSEC_DATA_COPY_INC_READERS_ATOMIC(  task->data[i].data_in );
                parsec_list_item_ring_chop((parsec_list_item_t*)task->data[i].data_in);
                PARSEC_LIST_ITEM_SINGLETON(task->data[i].data_in);
                parsec_list_push_back(&dealer_device->gpu_mem_owned_lru, (parsec_list_item_t*)task->data[i].data_in);
            }
            if( (PARSEC_FLOW_ACCESS_READ & gpu_task->flow[i]->flow_flags) &&
                (PARSEC_FLOW_ACCESS_WRITE & gpu_task->flow[i]->flow_flags)) 
            {
                assert(task->data[i].data_in->readers > 0);
                parsec_list_item_ring_chop((parsec_list_item_t*)task->data[i].data_in);
                PARSEC_LIST_ITEM_SINGLETON(task->data[i].data_in);
                parsec_list_push_back(&dealer_device->gpu_mem_owned_lru, (parsec_list_item_t*)task->data[i].data_in);

            }
            
            assert(task->data[i].data_in->original == task->data[i].data_out->original);
            assert( task->data[i].data_in->original != NULL);
            if( (PARSEC_FLOW_ACCESS_WRITE & gpu_task->flow[i]->flow_flags)  )
                assert( task->data[i].data_out->version == task->data[i].data_in->version);
            assert(task->data[i].data_out != NULL);
            assert(original->device_copies[dealer_device->super.device_index]!= NULL);
            assert(original->device_copies[dealer_device->super.device_index] == task->data[i].data_out);
            assert(task->data[i].data_in->readers >= 0);
            assert( task->data[i].data_out->version == task->data[i].data_in->version);
            if(task->data[i].data_out->original->owner_device != dealer_device->super.device_index)
                assert(task->data[i].data_out->version == task->data[i].data_in->version);
            assert(task->data[i].data_in->device_index == dealer_device->super.device_index);

            parsec_atomic_unlock( &original->lock );  


        }
        /**
         * Data is not yet staged in the dealer device, but some of the data we need maybe
         * already in the delaer device and the dealer device may have the latest data. 
         * In that case we set data_in = data_out.
         */
        else 
        {
            if(task->data[i].data_out->original->owner_device == dealer_device->super.device_index &&
                (task->data[i].data_out->version != task->data[i].data_out->original->device_copies[0]->version) )
            {
                parsec_data_t* original = task->data[i].data_out->original;
                parsec_atomic_lock( &original->lock );
                task->data[i].data_in = task->data[i].data_out;
                task->data[i].data_in->coherency_state = PARSEC_DATA_COHERENCY_SHARED;
                PARSEC_DATA_COPY_INC_READERS_ATOMIC(  task->data[i].data_in );
                PARSEC_OBJ_RETAIN(task->data[i].data_in);
                parsec_atomic_unlock( &original->lock );
            }
        }
    }

    return 0;
}


int gpu_data_version_increment(parsec_gpu_task_t *gpu_task, parsec_device_gpu_module_t* gpu_device)
{
    int i;
    parsec_task_t *task = gpu_task->ec;

    for(i = 0; i < task->task_class->nb_flows; i++)
    {
        if (task->data[i].data_out == NULL)
            continue;
        if(PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & gpu_task->flow[i]->flow_flags)) //CTL flow
            continue;

        if( (PARSEC_FLOW_ACCESS_WRITE & gpu_task->flow[i]->flow_flags) 
            && (gpu_task->task_type != PARSEC_GPU_TASK_TYPE_PREFETCH) ) 
        {
            parsec_gpu_data_copy_t* gpu_elem = task->data[i].data_out;
            gpu_elem->version++;  /* on to the next version */
            PARSEC_DEBUG_VERBOSE(10, parsec_gpu_output_stream,
                             "GPU[%s]: GPU copy %p [ref_count %d] increments version to %d at %s:%d",
                             gpu_device->super.name,
                             gpu_elem, gpu_elem->super.super.obj_reference_count, gpu_elem->version,
                             __FILE__, __LINE__);
        }
    }

    return 0;
}


