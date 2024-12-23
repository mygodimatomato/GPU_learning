#include <stdio.h>              // for printf (if you want to print)
#include <math.h>               // for sqrt, pow
#include <cuda_runtime.h>       // main CUDA runtime API
#include <device_launch_parameters.h>  // needed on some compilers for block/thread indexing

// Example array size: 1M elements
const int N = 1 << 20;  // 1,048,576

// Simple kernel that writes sqrt(pi^i) for each index
__global__ void kernel(float *x, int n)
{
    // Compute global thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Stride loop to cover all elements
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        // For numeric stability and performance, cast constants to float
        x[i] = sqrtf(powf(3.14159f, (float)i));
    }
}
int main()
{
    const int num_streams = 8;

    cudaStream_t streams[num_streams];
    float *data[num_streams];

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
 
        cudaMalloc(&data[i], N * sizeof(float));
        
        // launch one worker kernel per stream
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N);

        // launch a dummy kernel on the default stream
        kernel<<<1, 1>>>(0, 0);
    }

    cudaDeviceReset();

    return 0;
}