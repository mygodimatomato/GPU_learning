#Stream #CUDA 
Ref : https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/

## Introduction
- 這篇文章主要討論 CUDA Stream 的 default stream 造成的 blocking 行為
- CUDA 的 default stream 會導致 blocking, 但是在 CUDA 7 後可以算是緩和這個現象(下面提了兩個方法)
## Asynchronous Commands in CUDA
- 這段先提到了有幾個 CUDA command 是 asynchronous command (host 呼叫後就會繼續執行下去)
	- Kernel launch
	- Device-to-device copy
	- Host-to-device copy(with memory block 64 KB or less)
	- Async copy (using suffix "Async")
	- Memory set (using `cudaMemset`)
## The Default Stream
- 在 CUDA 7 之前, 每個 device 只有一個 default stream, 又因為 default stream 的特性(Default stream launch 時不能有其他 stream 在執行), 所以導致 performance 下降
- 但是在 CUDA 7 之後引入了 **per-thread default stream** 的概念, 其有兩個特性 : 
	1. <font color="#de7802">每個 host thread 都有自己的 default stream</font>
	2. <font color="#de7802">這些 default stream 會被視為 regular streams, 所以可以 concurrent 執行</font>
- 可以用 : `--default-stream per-thread` 或 `#define CUDA_API_PER_THREAD_DEFAULT_STREAM` (需要放在 `include cuda.h` 之前, 而且不能用在 .cu file 裡)

## A Multi-Stream Example
這裡先提了一個例子, 這個例子裡只有一個 thread 與多個 stream, 所以他只會有第二個 attribute, 也就是 default stream 會被視為 regular streams.
code 在 `stream_test.cu` 中, 如果是正常的 compile : 
``` bash
nvcc ./stream_test.cu -o stream_legacy
	``` 
- 則執行後的結果就會如下圖 : ![[Pasted image 20241223132036.png]]
- 可以看到 default stream 會 block  non-default stream 導致 performance degrade

接下來作者展示了 per-thread default stream 做法 : 
``` bash
	nvcc --default-stream per-thread ./stream_test.cu -o stream_per-thread	
```
-  則其執行結果會如下圖 : ![[Pasted image 20241223132926.png]]
- 可以看到 stream 17 是 default stream, 並且後續在 launch 時其他 stream 也可以執行, 因此 overlap 的程度上升, 整體 performance 提升

## A Multi-threading Example
接著作者提了在 pthread 的前提下, CUDA 7 可以提升的 performance, code 在 `pthread_test.cu` 如果是正常的 compile : 
``` bash
nvcc ./pthread_test.cu -o pthreads_legacy
```
- 則執行後的結果會如下圖 : ![[Pasted image 20241223134523.png]]
- 可以看到每個 thread 都是 sequential 的執行, 而跟我們想像中的執行方法有大差異.

接著則是 per-thread default stream 做法 : 
``` bash
nvcc --default-stream per-thread ./pthread_test.cu -o pthreads_per_thread
```
- 執行結果如下 : ![[Pasted image 20241223134737.png]]
- 整體平行度會拉高很多, 不再會有多個 default stream 互 block 的現象發生.

## More Tips
- `cudaDeviceSynchronize()` 也會 sync per-thread default stream, 如果要 sync single stream, 可以用 `cudaStreamSynchronize(cudaStream_t stream)`
- 也可以藉由呼叫 non-blocking streams 來達到類似的目的 (`cudaStreamCreate()` with `cudaStreamNonBlocking` flag) 
	- 差異在哪？non-blocking stream solution 的 default stream 會卡其他 stream 的 launch time
## Takeaway
- **per-thread default stream** have two attribute : 
	1. Each host thread gets its own default stream
	2. Default stream doesn’t block other streams
- use `--default-steram per-thread` to enable it 
- <font color="#de7802">This still works in CUDA 12 </font>
