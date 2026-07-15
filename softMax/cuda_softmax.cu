#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_cuda(float* data, float* output, int width)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float totalSum;
	if (threadIdx.x == 0)
	{
		totalSum = 0;
	}
	__syncthreads();
	float expV;
	if (tid < width)
	{	
		expV = expf (data [tid]);
		atomicAdd(&totalSum, expV);
	}
	__syncthreads();
	if (tid < width)
	{			
		output[tid] = expV / totalSum;
	}	
}