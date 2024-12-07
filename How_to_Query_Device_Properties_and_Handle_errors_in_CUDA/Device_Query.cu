#include <stdio.h> 

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Total Global Memory (GB): %f\n",
           prop.totalGlobalMem/1.0e9);
    printf("  Shared Memory per Block (KB): %d\n",prop.sharedMemPerBlock/1024);
    printf("  Registers per Block: %d\n", prop.regsPerBlock);
    printf("  Warp Size: %d\n", prop.warpSize);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max Threads per Block Dimension: %d x %d x %d\n",
           prop.maxThreadsDim[0],
           prop.maxThreadsDim[1],
           prop.maxThreadsDim[2]);

  }
}