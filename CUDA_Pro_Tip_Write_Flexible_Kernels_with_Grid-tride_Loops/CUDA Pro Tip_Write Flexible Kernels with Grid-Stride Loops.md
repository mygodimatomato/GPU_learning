#CUDA #Coding 

Ref : https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/

- 這篇主要在討論 CUDA code 的 scalability, 以及 optimization. 藉由 Grid-Stride 寫法可以提昇 CUDA code 的 portability, performance 以及 scalability, 算是一個很重要的 coding style
## 一般版本的 saxpy function ***(monolithic kernel)***: 
``` c
__global__
void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) 
        y[i] = a * x[i] + y[i];
}

// Perform SAXPY on 1M elements
saxpy<<<4096,256>>>(1<<20, 2.0, x, y);
```

## Grid-stride 版本的 saxpy function (code 在 github):
``` C
__global__
void saxpy(int n, float a, float *x, float *y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) 
      {
          y[i] = a * x[i] + y[i];
      }
}

int numSMs;
cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
// Perform SAXPY on 1M elements
saxpy<<<32*numSMs, 256>>>(1 << 20, 2.0, x, y);
```
- 假設有 1280 個 thread, 這個寫法中 thread 0 會 access element 0, 1280. 2560, ..., 這樣可以最大化 memory coalescing, 整體 performance 會上升
- 這個寫法還有以下幾個好處 : 
	1. Scalability && thread reuse : 可以 base on # of SM 來決定要 fire 多少個 thread.
	2. Debugging : `saxpy<<<1,1>>>(1<<20, 2.0, x, y);` 只要寫成這樣 code 還是可以跑, 而且會變成 serialize 的跑法, 更容易 debug.
	3. Portability, readability : 這點作者是提到可以用 Hemi library, 但是我其實不太清楚這有多常用.

- 以下是 profiling 的結果 : 
	- Gird_stride : ![[Screenshot 2024-12-09 at 1.36.18 PM.png]]
	- Original : ![[Screenshot 2024-12-09 at 1.36.32 PM.png]]
	- 可以看到整體 performance 幾乎相同, 甚至是更好, 以後要盡量用這種 coding style