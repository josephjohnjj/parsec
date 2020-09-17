/**
 * Copyright (c) 2019-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec.h"
#include "parsec/mca/device/cuda/device_cuda_internal.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/execution_stream.h"
#include "parsec/class/info.h"

#if defined(PARSEC_HAVE_CUDA)
#include <cublas_v2.h>
#endif

#include "nvlink.h"

#if defined(PARSEC_HAVE_CUDA)
static void destruct_cublas_handle(void *p)
{
    cublasHandle_t handle = (cublasHandle_t)p;
    cublasStatus_t status;
    if(NULL != handle) {
        status = cublasDestroy(handle);
        assert(status == CUBLAS_STATUS_SUCCESS);
    }
}

static void *create_cublas_handle(void *obj, void *p)
{
    cublasHandle_t handle;
    cublasStatus_t status;
    parsec_gpu_exec_stream_t *stream = (parsec_gpu_exec_stream_t *)obj;
    (void)p;
    /* No need to call cudaSetDevice, as this has been done by PaRSEC before calling the task body */
    status = cublasCreate(&handle);
    assert(CUBLAS_STATUS_SUCCESS == status);
    status = cublasSetStream(handle, stream->cuda_stream);
    assert(CUBLAS_STATUS_SUCCESS == status);
    return (void*)handle;
}
#endif

static void destroy_cublas_handle(void *_h, void *_n)
{
#if defined(PARSEC_HAVE_CUDA)
    cublasHandle_t cublas_handle = (cublasHandle_t)_h;
    cublasDestroy_v2(cublas_handle);
#endif
    (void)_n;
    (void)_h;
}

parsec_taskpool_t* testing_nvlink_New( parsec_context_t *ctx, int depth, int mb )
{
    parsec_nvlink_taskpool_t* testing_handle = NULL;
    int *dev_index, nb, dev, i;
    two_dim_block_cyclic_t *dcA;

    /** Find all CUDA devices */
    nb = 0;
    for(dev = 0; dev < (int)parsec_nb_devices; dev++) {
        parsec_device_module_t *device = parsec_mca_device_get(dev);
        if( PARSEC_DEV_CUDA == device->type ) {
            nb++;
        }
    }
    if(nb == 0) {
        char hostname[256];
        gethostname(hostname, 256);
        fprintf(stderr, "This test requires at least one CUDA device per node -- no CUDA device found on rank %d on %s\n",
                ctx->my_rank, hostname);
        return NULL;
    }
    dev_index = (int*)malloc(nb * sizeof(int));
    nb = 0;
    for(dev = 0; dev < (int)parsec_nb_devices; dev++) {
        parsec_device_module_t *device = parsec_mca_device_get(dev);
        if( PARSEC_DEV_CUDA == device->type ) {
            dev_index[nb++] = device->device_index;
        }
    }

#if defined(PARSEC_HAVE_CUDA)
    parsec_info_id_t CuHI = parsec_info_register(&parsec_per_stream_infos, "CUBLAS::HANDLE",
                                                 destroy_cublas_handle, NULL,
                                                 create_cublas_handle, NULL,
                                                 NULL);
    assert(CuHI != -1);
#else
    int CuHI = -1;
#endif

    dcA = (two_dim_block_cyclic_t*)calloc(1, sizeof(two_dim_block_cyclic_t));
    two_dim_block_cyclic_init(dcA, matrix_RealDouble, matrix_Tile,
                              ctx->my_rank,
                              mb, mb,
                              depth*mb, ctx->nb_nodes*mb,
                              0, 0,
                              depth*mb, ctx->nb_nodes*mb,
                              1, ctx->nb_nodes, 1, 1,
                              0, 0);
    dcA->mat = parsec_data_allocate((size_t)dcA->super.nb_local_tiles *
                                    (size_t)dcA->super.bsiz *
                                   (size_t)parsec_datadist_getsizeoftype(dcA->super.mtype));
    parsec_data_collection_set_key((parsec_data_collection_t*)dcA, "A");

    for(i = 0; i < dcA->super.nb_local_tiles * mb * mb; i++)
        ((double*)dcA->mat)[i] = (double)rand() / (double)RAND_MAX;

    testing_handle = parsec_nvlink_new(dcA, ctx->nb_nodes, CuHI, nb, dev_index);

    parsec_matrix_add2arena( &testing_handle->arenas_datatypes[PARSEC_nvlink_DEFAULT_ADT_IDX],
                             parsec_datatype_double_complex_t,
                             matrix_UpperLower, 1, mb, mb, mb,
                             PARSEC_ARENA_ALIGNMENT_SSE, -1 );

    return &testing_handle->super;
}

void testing_nvlink_Destruct( parsec_taskpool_t *tp )
{
    parsec_nvlink_taskpool_t *nvlink_taskpool = (parsec_nvlink_taskpool_t *)tp;
    two_dim_block_cyclic_t *dcA;
    parsec_matrix_del2arena( & nvlink_taskpool->arenas_datatypes[PARSEC_nvlink_DEFAULT_ADT_IDX] );
    parsec_data_free(nvlink_taskpool->_g_descA->mat);
    parsec_info_unregister(&parsec_per_stream_infos, nvlink_taskpool->_g_CuHI, NULL);
    dcA = nvlink_taskpool->_g_descA;
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)nvlink_taskpool->_g_descA );
    parsec_taskpool_free(tp);
    free(dcA);
}
