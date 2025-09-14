#Unified_Memory #GPU #Basic 
Ref : https://developer.nvidia.com/blog/even-easier-introduction-cuda/

## Starting Simple
- sequential version 的 vector add → code 在 `add.cpp` 中

## Memory Allocation in CUDA
- 這個版本是用 1 個 thread 來 run all vector add → code 在 `add.cu` 中
- 這裡使用 unified memory 來實作
	- `cudaMallocManaged()` : return a pointer that you can access from host code or device code
	- `cudaFree()` : free the data
	- `cudaDeviceSynchronize()` : 確保 synchronization, 放在 final error checking on the CPU
- profiling output : ![[Screenshot 2024-12-02 at 12.14.16 AM.png]]
	- profiling 的重點有 `GPU activities:  100.00%  2.8207ms `
	
## Picking up the Threads
- 這個版本是用一個 block, 256 個 thread 來 run vector add → code 在 `add_block.cu` 中
- profiling output : ![[Screenshot 2024-12-02 at 12.19.19 AM.png]]
	- profiling 可以看到從原本的 210 ms → 2.9 ms

## Out of the Blocks
- 這個版本是用 grid-stride loop 來實作 → code 在 `add_grid.cu` 中
- 這個實作方法宣告了 256 個 thread per block, 並宣告了 4096 個 block, 以此來增加這份 code 的 scalability. 以下是示意圖 : ![[Screenshot 2024-12-02 at 12.23.27 AM.png]]

- profiling output : ![[Screenshot 2024-12-02 at 12.21.52 AM.png]]
- profiling output 可以看到 performance 沒有什麼提升 (都是 3ms), 但是作者文章中的 performance 卻插到 28 倍, 詳細原因如下 : 
	- I'm glad you asked! The short answer is because on pre-pascal GPUs that can't page fault, the data that has been touched on the CPU has to be copied (automatically) to the GPU before the kernel launches. But that copy is not included in the timing of the kernel in the profiler. On Pascal, only the data that is touched by the GPU is migrated, and it happens on demand (at the time of a GPU page fault) -- in this case though that means all the data. The cost of the page faults and migrations therefore impacts the runtime of the kernel on Pascal. If you run the kernel twice, though, you'll see that the second run is faster (because there are no page faults -- the data is already all on the GPU). There are other options, like initializing the data on the GPU. I'm preparing a follow up post which goes over this in some detail. See [https://devblogs.nvidia.com...](https://devblogs.nvidia.com/parallelforall/cuda-8-features-revealed/ "https://devblogs.nvidia.com/parallelforall/cuda-8-features-revealed/") and  [https://devblogs.nvidia.com...](https://devblogs.nvidia.com/parallelforall/beyond-gpu-memory-limits-unified-memory-pascal/ "https://devblogs.nvidia.com/parallelforall/beyond-gpu-memory-limits-unified-memory-pascal/")
	- <span style="background:#b1ffff">不整理的原因是我還在想這到底對不對, 理論上會在下一篇裡面提到</span> → [下一篇](obsidian://open?vault=Notes&file=Work%2FNvidia%2FGPU_Learning%2FUnified%20Memory%20for%20CUDA%20Beginners)確實有提到

## Exercises
1. Experiment with [`printf()`](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#formatted-output) inside the kernel. Try printing out the values of `threadIdx.x`and `blockIdx.x` for some or all of the threads. Do they print in sequential order? Why or why not?
2. Print the value of `threadIdx.y` or `threadIdx.z` (or `blockIdx.y`) in the kernel. (Likewise for `blockDim` and `gridDim`). Why do these exist? How do you get them to take on values other than 0 (1 for the dims)?
3. If you have access to a [Pascal-based GPU](https://developer.nvidia.com/blog/inside-pascal/), try running `add_grid.cu` on it. Is performance better or worse than the K80 results? Why? (Hint: read about [Pascal’s Page Migration Engine and the CUDA 8 Unified Memory API](https://developer.nvidia.com/blog/beyond-gpu-memory-limits-unified-memory-pascal/).) For a detailed answer to this question, see the post [Unified Memory for CUDA Beginners](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/). #TODO