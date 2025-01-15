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

    int device = 0;
    cudaGetDevice(&device);

    // Create separate events for prefetch and kernel timing
    cudaEvent_t Start, Stop;
    cudaEventCreate(&Start);
    cudaEventCreate(&Stop);

    float totalTime = 0.0f;
    const int numIterations = 100;

    cudaEventRecord(Start);
    for (int i = 0; i < numIterations; ++i) {
        for (size_t j = 0; j < 6710864 * 8; ++j) {
            managedPtr[j] = static_cast<float>(j+i);
        }
        cudaMemPrefetchAsync(managedPtr, dataSize, device);

        stream_thread<float, READ><<<1, 256>>>(managedPtr, dataSize, output, 0.0f);
    }
    cudaEventRecord(Stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&totalTime, Start, Stop);
    // Calculate averages
    float avgTime = totalTime / numIterations;

    std::cout << "Average Time: " << avgTime << " ms" << std::endl;

    cudaFree(managedPtr);
    cudaFree(output);
    cudaEventDestroy(Start);
    cudaEventDestroy(Stop);

    return 0;
}
