#include <cuda_runtime.h>
#include <iostream>

#define READ 0
#define WRITE 1

template <typename data_type, int op>
__global__ void stream_thread(data_type *ptr, const size_t size, 
                              data_type *output, const data_type val) 
{ 
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x; 
  size_t n = size / sizeof(data_type); 
  data_type accum = 0; 

  for(; tid < n; tid += blockDim.x * gridDim.x) 
    if (op == READ) accum += ptr[tid]; 
      else ptr[tid] = val;  

  if (op == READ) 
    output[threadIdx.x + blockIdx.x * blockDim.x] = accum; 
}

int main() {
    const size_t dataSize = 6710864 * 8 * sizeof(float); // Size in bytes
    float *managedPtr, *output;
    cudaMallocManaged(&managedPtr, dataSize);
    cudaMallocManaged(&output, dataSize);

    for (size_t i = 0; i < 6710864 * 8; ++i) {
        managedPtr[i] = static_cast<float>(i);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Measure kernel execution time
    cudaEventRecord(start);
    stream_thread<float, READ><<<1, 256>>>(managedPtr, dataSize, output, 0.0f);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float kernelTime = 0.0f;
    cudaEventElapsedTime(&kernelTime, start, stop);

    // Calculate bandwidth
    float bandwidth = (dataSize / kernelTime) / 1e6; // GB/s

    std::cout << "On-demand Migration - Kernel Bandwidth: " << bandwidth << " GB/s" << std::endl;

    cudaFree(managedPtr);
    cudaFree(output);
    return 0;
}
