#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>

#define MAX_BLOCKS 65535

__global__ void device_copy_vector4_kernel(int* d_in, int* d_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N/4; i += blockDim.x * gridDim.x) {
        reinterpret_cast<int4*>(d_out)[i] = reinterpret_cast<int4*>(d_in)[i];
    }

    // in only one thread, process final element(if there is one)
    int remainder = N % 4;
    if (idx==N/4 && remainder != 0) {
      while(remainder) {
        int idx = N - remainder--;
        d_out[idx] = d_in[idx];
      }
    }
}

void device_copy_vector4(int* d_in, int* d_out, int N) {
    int threads = 128;
    int blocks = min((N + threads - 1) / threads, MAX_BLOCKS);
    // int blocks = (N + threads - 1) / threads;

    // Measure kernel execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    device_copy_vector4_kernel<<<blocks, threads>>>(d_in, d_out, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate and print bandwidth
    float gb = (2.0f * N * sizeof(int)) / (1e9); // 2xN because both input and output are accessed
    float bandwidth = gb / (milliseconds / 1000.0f); // GB/s
    std::cout << "Array size: " << N << ", Kernel execution time: " << milliseconds << " ms, Bandwidth: " << bandwidth << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    std::vector<int> sizes;
    for (int size = 4096; size <= 536870912; size *= 2) {
        sizes.push_back(size);
    }

    for (int N : sizes) {
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

        // Measure data transfer time from host to device
        auto start_h2d = std::chrono::high_resolution_clock::now();
        cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
        auto end_h2d = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> h2d_time = end_h2d - start_h2d;

        std::cout << "Array size: " << N << ", Host to Device transfer time: " << h2d_time.count() << " seconds" << std::endl;

        // Call the device copy function
        device_copy_vector4(d_in, d_out, N);

        // Measure data transfer time from device to host
        auto start_d2h = std::chrono::high_resolution_clock::now();
        cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
        auto end_d2h = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> d2h_time = end_d2h - start_d2h;

        std::cout << "Array size: " << N << ", Device to Host transfer time: " << d2h_time.count() << " seconds" << std::endl;

        // Verify the output
        bool success = true;
        for (int i = 0; i < N; i++) {
            if (h_out[i] != h_in[i]) {
                success = false;
                break;
            }
        }

        if (success) {
            std::cout << "Array size: " << N << ", Data copied successfully!" << std::endl;
        } else {
            std::cout << "Array size: " << N << ", Data copy failed." << std::endl;
        }

        // Free device and host memory
        cudaFree(d_in);
        cudaFree(d_out);
        delete[] h_in;
        delete[] h_out;
    }

    return 0;
}
