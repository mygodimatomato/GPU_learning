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

    // for (size_t i = 0; i < 6710864 * 8; ++i) {
    //     managedPtr[i] = static_cast<float>(i);
    // }

    int device = 0;
    cudaGetDevice(&device);

    // Create separate events for prefetch and kernel timing
    cudaEvent_t prefetchStart, prefetchStop;
    cudaEvent_t kernelStart, kernelStop;
    cudaEventCreate(&prefetchStart);
    cudaEventCreate(&prefetchStop);
    cudaEventCreate(&kernelStart);
    cudaEventCreate(&kernelStop);

    float totalPrefetchTime = 0.0f;
    float totalKernelTime = 0.0f;
    const int numIterations = 100;

    for (int i = 0; i < numIterations; ++i) {
        for (size_t j = 0; j < 6710864 * 8; ++j) {
            managedPtr[j] = static_cast<float>(j+i);
        }
        // Measure prefetching time
        cudaEventRecord(prefetchStart);
        cudaMemPrefetchAsync(managedPtr, dataSize, device);
        cudaEventRecord(prefetchStop);
        cudaDeviceSynchronize();

        float prefetchTime = 0.0f;
        cudaEventElapsedTime(&prefetchTime, prefetchStart, prefetchStop);
        totalPrefetchTime += prefetchTime;

        // Measure kernel execution time
        cudaEventRecord(kernelStart);
        stream_thread<float, READ><<<1, 256>>>(managedPtr, dataSize, output, 0.0f);
        cudaEventRecord(kernelStop);
        cudaDeviceSynchronize();

        float kernelTime = 0.0f;
        cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop);
        totalKernelTime += kernelTime;
    }

    // Calculate averages
    float avgPrefetchTime = totalPrefetchTime / numIterations;
    float avgKernelTime = totalKernelTime / numIterations;

    // Calculate average bandwidth
    float avgBandwidth = (dataSize / (avgPrefetchTime + avgKernelTime)) / 1e6; // GB/s

    std::cout << "Average Prefetch Time: " << avgPrefetchTime << " ms" << std::endl;
    std::cout << "Average Kernel Execution Time: " << avgKernelTime << " ms" << std::endl;
    std::cout << "Average Bandwidth: " << avgBandwidth << " GB/s" << std::endl;

    cudaFree(managedPtr);
    cudaFree(output);
    cudaEventDestroy(prefetchStart);
    cudaEventDestroy(prefetchStop);
    cudaEventDestroy(kernelStart);
    cudaEventDestroy(kernelStop);

    return 0;
}
