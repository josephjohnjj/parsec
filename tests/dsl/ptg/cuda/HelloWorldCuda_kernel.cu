#include <stdio.h>

extern "C" 
{
    void HelloWorld_cuda_kernel(double *A_double, int k);
}

__global__ void cuda_kernel(double *A_double, int k) 
{
    int i;
    for( i = 0; i < 100; i++ )
        *(A_double+i) = *(A_double+i) + k; 
}

void HelloWorld_cuda_kernel(double *A_double, int k)
{
    cuda_kernel<<<1,1>>>(A_double, k);
}