#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

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
    const size_t dataSize = 1024 * sizeof(float);
    float *hostPtr, *devicePtr, *output;
    cudaMallocHost(&hostPtr, dataSize);  // Allocate pinned host memory
    cudaMalloc(&devicePtr, dataSize);   // Allocate GPU memory
    cudaMalloc(&output, dataSize);

    // Initialize host memory
    for (size_t i = 0; i < 1024; ++i) {
        hostPtr[i] = static_cast<float>(i);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Measure data transfer time
    cudaEventRecord(start);
    cudaMemcpyAsync(devicePtr, hostPtr, dataSize, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float transferTime = 0.0f;
    cudaEventElapsedTime(&transferTime, start, stop);

    // Measure kernel execution time
    cudaEventRecord(start);
    stream_thread<float, READ><<<1, 256>>>(devicePtr, dataSize, output, 0.0f);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float kernelTime = 0.0f;
    cudaEventElapsedTime(&kernelTime, start, stop);

    std::cout << "Explicit Data Transfer - Transfer Time: " << transferTime << " ms, Kernel Time: " << kernelTime << " ms" << std::endl;
    std::cout << "Total bandwidth: " << (dataSize / (transferTime + kernelTime)) / 1e6 << " GB/s" << std::endl; 

    cudaFreeHost(hostPtr);
    cudaFree(devicePtr);
    cudaFree(output);
    return 0;
}
