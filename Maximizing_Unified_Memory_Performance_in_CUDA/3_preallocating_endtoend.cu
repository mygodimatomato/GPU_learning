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
    const int numIterations = 100; // Number of repetitions
    float *hostPtr, *devicePtr, *output;
    cudaMallocHost(&hostPtr, dataSize);  // Allocate pinned host memory
    cudaMalloc(&devicePtr, dataSize);   // Allocate GPU memory
    cudaMalloc(&output, dataSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < numIterations; ++i) {
        for (size_t j = 0; j < 6710864 * 8; ++j) {
            hostPtr[j] = static_cast<float>(j+i);
        }
        cudaMemcpyAsync(devicePtr, hostPtr, dataSize, cudaMemcpyHostToDevice);
        stream_thread<float, READ><<<1, 256>>>(devicePtr, dataSize, output, 0.0f);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float totalTime = 0.0f;
    cudaEventElapsedTime(&totalTime, start, stop);

    float averageTime = totalTime / numIterations;

    std::cout << "Average Time: " << averageTime << " ms" << std::endl;


    cudaFreeHost(hostPtr);
    cudaFree(devicePtr);
    cudaFree(output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
