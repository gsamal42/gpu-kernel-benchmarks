#include <torch/extension.h>
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


__global__ void softmax_Handle_INF_cuda(float* data, float* output, int width)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float totalSum;
	__shared__ float maxValue;
	if (threadIdx.x == 0)
	{
		totalSum = 0;
		maxValue = data[0];
	}
	__syncthreads();
	if (tid < width)
	{	
		for (int i = 1; i < width; i++) {
            maxValue = max(maxValue, data [tid]);
        }
	}
	__syncthreads();
	float expV;
	if (tid < width)
	{	
		expV = expf (data [tid] - maxValue);
		atomicAdd(&totalSum, expV);
	}
	__syncthreads();
	if (tid < width)
	{			
		output[tid] = expV / totalSum;
	}	
}

torch::Tensor softmax_forward(
    torch::Tensor input)
{
    auto output =
        torch::zeros_like(input);

    int width =
        input.numel();

    softmax_cuda<<<1, width>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        width);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward",
          &softmax_forward,
          "SoftMax forward");
}