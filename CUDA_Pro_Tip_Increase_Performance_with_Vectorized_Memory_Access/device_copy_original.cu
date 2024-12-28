#include <cuda_runtime.h>
#include <iostream>

#define MAX_BLOCKS 65535

__global__ void device_copy_scalar_kernel(int* d_in, int* d_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        d_out[i] = d_in[i];
    }
}

void device_copy_scalar(int* d_in, int* d_out, int N) {
    int threads = 128;
    int blocks = min((N + threads - 1) / threads, MAX_BLOCKS);
    device_copy_scalar_kernel<<<blocks, threads>>>(d_in, d_out, N);
}

int main() {
    const int N = 4096;
    int *h_in = new int[N];
    int *h_out = new int[N];

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_in[i] = i;
    }

    // Device memory pointers
    int *d_in, *d_out;

    // Allocate device memory
    cudaMalloc((void**)&d_in, N * sizeof(int));
    cudaMalloc((void**)&d_out, N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // Call the device copy function
    device_copy_scalar(d_in, d_out, N);

    // Copy data back from device to host
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify the output
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_in[i]) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Data copied successfully!" << std::endl;
    } else {
        std::cout << "Data copy failed." << std::endl;
    }

    // Free device and host memory
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}
