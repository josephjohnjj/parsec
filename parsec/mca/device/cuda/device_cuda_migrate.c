#include "parsec/mca/device/cuda/device_cuda_migrate.h"

extern int parsec_device_cuda_enabled;
parsec_device_cuda_info_t* device_info; 
static parsec_list_t* migrated_task_list;
static int NDEVICES;
migration_accounting_t* accounting;
static parsec_hash_table_t *migrated_data_hash_table = NULL;


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
        accounting[i].received = 0;
    }

    #if defined(PARSEC_HAVE_CUDA)
    nvml_ret = nvmlInit_v2();
    #endif

    migrated_data_hash_table = PARSEC_OBJ_NEW(parsec_hash_table_t); 
    parsec_hash_table_init(migrated_data_hash_table,
                           offsetof(migrated_data_t, ht_item),
                           8, migrated_data_key_fns, NULL);
    

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
        printf("Task received %d \n", accounting[i].received);
    }
    PARSEC_OBJ_RELEASE(migrated_task_list); 
    free(device_info); 

    parsec_hash_table_fini(migrated_data_hash_table);

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
    parsec_gpu_task_t *migrated_gpu_task = NULL;
    parsec_device_gpu_module_t* dealer_device = NULL;
    parsec_device_gpu_module_t* starving_device = NULL;
    int stage_in_status = 0;


    mig_task = (migrated_task_t*) parsec_list_pop_front(migrated_task_list);

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
        parsec_atomic_fetch_inc_int32(&accounting[CUDA_DEVICE_NUM(starving_device->super.device_index)].received); 
        parsec_cuda_kernel_scheduler(es, (parsec_gpu_task_t *) migrated_gpu_task, starving_device->super.device_index);  
	    PARSEC_OBJ_DESTRUCT(mig_task);
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

int migrate_if_starving(parsec_execution_stream_t *es,  parsec_device_gpu_module_t* dealer_device)
{
    int starving_device_index = -1, dealer_device_index = 0;
    int nb_migrated = 0, execution_level = 0, stream_index = 0, j = 0;
    parsec_gpu_task_t *migrated_gpu_task = NULL;
    parsec_device_gpu_module_t* starving_device = NULL;
    migrated_task_t *mig_task = NULL;

    dealer_device_index = CUDA_DEVICE_NUM(dealer_device->super.device_index);  
    if(is_starving(dealer_device_index))
        return 0;
    
    starving_device_index = find_starving_device(dealer_device_index);
    if(starving_device_index == -1)
        return 0;
    starving_device = (parsec_device_gpu_module_t*)parsec_mca_device_get(DEVICE_NUM(starving_device_index));

    //#if 0
    migrated_gpu_task = (parsec_gpu_task_t*)parsec_list_pop_front( &(dealer_device->pending) ); //level 0
    execution_level = 0;
    if(migrated_gpu_task == NULL)
    {
        migrated_gpu_task = (parsec_gpu_task_t*)parsec_list_pop_front( dealer_device->exec_stream[0]->fifo_pending ); //level 1
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
    //#endif

    //for(j = 0; j < (dealer_device->max_exec_streams - 2); j++)
    //{
    //    migrated_gpu_task = (parsec_gpu_task_t*)parsec_list_try_pop_back( dealer_device->exec_stream[ (2 + j) ]->fifo_pending ); //level2
    //    if(migrated_gpu_task != NULL)
    //    {
    //        execution_level = 2;
    //        stream_index = 2 + j;
    //        break;
    //    }
    //}
    

    if(migrated_gpu_task != NULL)
    {
	    assert(migrated_gpu_task->ec != NULL);
        parsec_list_item_ring_chop( (parsec_list_item_t*)migrated_gpu_task );
        PARSEC_LIST_ITEM_SINGLETON((parsec_list_item_t*)migrated_gpu_task);
        /**
         * @brief if the GPU task is a not a computational kerenel
         * stop migration.
         */
        if(migrated_gpu_task->task_type != PARSEC_GPU_TASK_TYPE_KERNEL || migrated_gpu_task->migrate_status > TASK_NOT_MIGRATED)
        {
            if(execution_level == 0)
            {
                parsec_list_push_front(&(dealer_device->pending), (parsec_list_item_t*) migrated_gpu_task );
            }
            if(execution_level == 1)
            {
                parsec_list_push_front( dealer_device->exec_stream[0]->fifo_pending, (parsec_list_item_t*) migrated_gpu_task );
            }
            if(execution_level == 2)
            {
                parsec_list_push_front( dealer_device->exec_stream[stream_index]->fifo_pending, (parsec_list_item_t*) migrated_gpu_task );
            }
            
            return nb_migrated;
        }

        assert( (migrated_gpu_task != NULL) && (migrated_gpu_task->ec != NULL) );

        if(execution_level == 0)
            accounting[CUDA_DEVICE_NUM(dealer_device->super.device_index)].level0++;
        if(execution_level == 1)
            accounting[CUDA_DEVICE_NUM(dealer_device->super.device_index)].level1++;
        if(execution_level == 2)
            accounting[CUDA_DEVICE_NUM(dealer_device->super.device_index)].level2++;
        nb_migrated++;
        parsec_cuda_set_device_task(dealer_device_index, /* count */ -1, /* level */ 0); // decrement task count at the dealer device

        //change migrate_status
        if( execution_level == 2 )
            migrated_gpu_task->migrate_status = TASK_MIGRATED_AFTER_STAGE_IN; 
        else
            migrated_gpu_task->migrate_status = TASK_MIGRATED_BEFORE_STAGE_IN;

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
    return nb_migrated;
}

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

        if( stage_in_status == TASK_MIGRATED_AFTER_STAGE_IN )
        {   
            parsec_data_t* original = task->data[i].data_out->original;
            parsec_atomic_lock( &original->lock );

            //staged in data is already available it data_out
            task->data[i].data_in = task->data[i].data_out;
            task->data[i].data_in->coherency_state = PARSEC_DATA_COHERENCY_SHARED;
            PARSEC_OBJ_RETAIN(task->data[i].data_in);

            /**
             * @brief If the task only has write access remove it from owned LRU and add
             * it to mem LRU. Increment the reader to make usre the data will 
             * not be evicted.  
             */
            if( !(PARSEC_FLOW_ACCESS_READ & gpu_task->flow[i]->flow_flags) &&
                (PARSEC_FLOW_ACCESS_WRITE & gpu_task->flow[i]->flow_flags)) 
            {
                assert(task->data[i].data_in->readers > 0);
                PARSEC_DATA_COPY_INC_READERS_ATOMIC(  task->data[i].data_in );
                parsec_list_item_ring_chop((parsec_list_item_t*)task->data[i].data_in);
                PARSEC_LIST_ITEM_SINGLETON(task->data[i].data_in);
                parsec_list_push_back(&dealer_device->gpu_mem_owned_lru, (parsec_list_item_t*)task->data[i].data_in);
            }
            /**
             * @brief For read_write the flow the readers will already be inceremnetd
             * but, the data will be in the owned LRU move it to the mem LRU
             */
            if( (PARSEC_FLOW_ACCESS_READ & gpu_task->flow[i]->flow_flags) &&
                (PARSEC_FLOW_ACCESS_WRITE & gpu_task->flow[i]->flow_flags)) 
            {
                //PARSEC_DATA_COPY_INC_READERS_ATOMIC(  task->data[i].data_in );
                assert(task->data[i].data_in->readers > 0);
                parsec_list_item_ring_chop((parsec_list_item_t*)task->data[i].data_in);
                PARSEC_LIST_ITEM_SINGLETON(task->data[i].data_in);
                parsec_list_push_back(&dealer_device->gpu_mem_owned_lru, (parsec_list_item_t*)task->data[i].data_in);

            }
            if( (PARSEC_FLOW_ACCESS_READ & gpu_task->flow[i]->flow_flags) &&
                !(PARSEC_FLOW_ACCESS_WRITE & gpu_task->flow[i]->flow_flags)) 
            {
                //PARSEC_DATA_COPY_INC_READERS_ATOMIC(  task->data[i].data_in );
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
        else 
        {
            /**
             * The data GPU will be the owner of the data, only if the task executing on that GPU
             * has a write permission on it. (data.c line 423)
             */
            if(task->data[i].data_out->original->owner_device != dealer_device->super.device_index)
                continue;
            //if( task->data[i].data_in->version == task->data[i].data_out->version)
            //    continue;
        //
            //task->data[i].data_in = task->data[i].data_out;


            if(task->data[i].data_out->original->owner_device != dealer_device->super.device_index)
                assert( task->data[i].data_out->version == task->data[i].data_out->original->device_copies[0]->version); 
            parsec_data_t* original = task->data[i].data_in->original;
            parsec_atomic_lock( &original->lock );

            //parsec_list_item_ring_chop((parsec_list_item_t*)task->data[i].data_in);
            //PARSEC_LIST_ITEM_SINGLETON(task->data[i].data_in);
            
            assert(task->data[i].data_in->readers >= 0);

            task->data[i].data_in->coherency_state = PARSEC_DATA_COHERENCY_SHARED;

            parsec_atomic_unlock( &original->lock );   
        }
    }

    return 0;
}

int gpu_data_compensate_reader(parsec_gpu_task_t *gpu_task, parsec_device_gpu_module_t *gpu_device)
{
    int i;
    parsec_task_t *task = gpu_task->ec;

    for(i = 0; i < task->task_class->nb_flows; i++)
    {
        if (task->data[i].data_in == NULL)
            continue;
        if(PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & gpu_task->flow[i]->flow_flags)) //CTL flow
            continue;
            
        parsec_atomic_lock( &task->data[i].data_in->original->lock );

        //PARSEC_DATA_COPY_DEC_READERS_ATOMIC( task->data[i].data_in );
        //PARSEC_OBJ_RELEASE(task->data[i].data_in);
        parsec_device_gpu_module_t *src_device =
                    (parsec_device_gpu_module_t*)parsec_mca_device_get( task->data[i].data_in->device_index );
        if(0 == task->data[i].data_in->readers)
        {
            assert( ((parsec_list_item_t*)task->data[i].data_in)->list_next != (parsec_list_item_t*)task->data[i].data_in );
            assert( ((parsec_list_item_t*)task->data[i].data_in)->list_prev != (parsec_list_item_t*)task->data[i].data_in );
            
            //parsec_list_item_ring_chop((parsec_list_item_t*)task->data[i].data_in);
            //PARSEC_LIST_ITEM_SINGLETON(task->data[i].data_in);
            //parsec_list_push_back(&src_device->gpu_mem_lru, (parsec_list_item_t*)task->data[i].data_in);
            
        }

        parsec_atomic_unlock( &task->data[i].data_in->original->lock );
        
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




int migrate_hash_table_insert( parsec_gpu_task_t *migrated_gpu_task, parsec_device_gpu_module_t* dealer_device)
{
    int i;
    migrated_data_t *migrated_data_item = NULL;
    parsec_task_t *task = migrated_gpu_task->ec;
    migrated_data_item = (migrated_data_t *) calloc(1, sizeof(migrated_data_t));
    migrated_data_item->dealer_device = dealer_device;
    migrated_data_item->ht_item.key = (parsec_key_t) task->task_class->make_key((const parsec_taskpool_t*)task->taskpool, 
                                                                                (const parsec_assignment_t*)&task->locals);
    for( i = 0; i < task->task_class->nb_flows; i++) 
    {     
        if(PARSEC_FLOW_ACCESS_NONE == (PARSEC_FLOW_ACCESS_MASK & migrated_gpu_task->flow[i]->flow_flags)) //CTL flow
        {
            migrated_data_item->old_copy[i] = NULL;
            continue;   
        }

        if(task->data->data_out == NULL)   
            migrated_data_item->old_copy[i] = NULL;
        else
            migrated_data_item->old_copy[i] = task->data->data_out;
    }

    parsec_hash_table_lock_bucket(migrated_data_hash_table, migrated_data_item->ht_item.key);                                                                                             
    parsec_hash_table_nolock_insert(migrated_data_hash_table, &migrated_data_item->ht_item);
    parsec_hash_table_unlock_bucket(migrated_data_hash_table, migrated_data_item->ht_item.key);

    return 1;
}

int migrate_hash_table_delete( parsec_gpu_task_t *migrated_gpu_task)
{
    int i;
    migrated_data_t* migrated_data_item = NULL;
    parsec_task_t* task = migrated_gpu_task->ec;
    parsec_key_t key;

    
    key = (parsec_key_t) migrated_gpu_task->ec->task_class->make_key((const parsec_taskpool_t*)migrated_gpu_task->ec->taskpool, 
                                                                     (const parsec_assignment_t*)&migrated_gpu_task->ec->locals);

    parsec_hash_table_lock_bucket(migrated_data_hash_table, key);                                                                                             
    migrated_data_item = (migrated_data_t*) parsec_hash_table_nolock_remove(migrated_data_hash_table, key);
    parsec_hash_table_unlock_bucket(migrated_data_hash_table, key);
    
    if( migrated_data_item != NULL)
    {
        if( migrated_gpu_task->migrate_status == TASK_MIGRATED_AFTER_STAGE_IN)
        {
            for( i = 0; i < task->task_class->nb_flows; i++)   
            {     
                if(migrated_data_item->old_copy[i] == NULL)   
                    continue;

                parsec_data_t* original = migrated_data_item->old_copy[i]->original;
                parsec_atomic_lock( &original->lock );

                if( (PARSEC_FLOW_ACCESS_READ & migrated_gpu_task->flow[i]->flow_flags) )
                {
                    PARSEC_DATA_COPY_DEC_READERS_ATOMIC(migrated_data_item->old_copy[i]);
                }

                parsec_list_push_back(&migrated_data_item->dealer_device->gpu_mem_lru, 
                    (parsec_list_item_t*)migrated_data_item->old_copy[i]);

                parsec_atomic_unlock( &original->lock );
            }
        }

        free(migrated_data_item);
    }


    return 1;
}
