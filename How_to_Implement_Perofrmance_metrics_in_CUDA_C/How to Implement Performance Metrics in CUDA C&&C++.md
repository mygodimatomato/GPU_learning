#CUDA #Profiling #Debug
Ref : https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/

## Timing Kernel Execution with CPU Timers
- 這段提到了一般的 CUDA function 是 non-blocking, 所以如果用 cpu-time 去算 execution time 會導致算錯
- 如果要用 cpu-time 去 check, 就需要用 `cudaDeviceSynchronizse()` 放在 kernel 後面, 才能算到正確的時間

## Timing using CUDA Events
- 如果用 `cudaDeviceSynchronize()`, 會導致 GPU pipeline stall, 所以可以用 `cudaEvent` 來計算時間
	- **`cudaEventRecord`** does not block. It simply places the event into the GPU's stream and associates it with all preceding operations in that stream. It essentially marks a point in the stream.
	- **`cudaEventSynchronize`**, on the other hand, blocks the host code until the GPU has completed all operations scheduled before the event was recorded.
	- `cudaEventSynchronize` 會等 parameter 中的那個 event 發生後才讓後續的 code 執行下去

## Memory Bandwidth
- Theoretical Bandwidth
	- 可以找到 P100-16GB 的 Theoretical Bandwidth 是 720GB
	- 算法是 : memory clock rate * memory interface * dual channel
-  Effective Bandwidth
	- $BW_{Effective} = (R_{B} + W_{B}) / (t * 10^9)$
	- $BW_{Effective}$ : Effective bandwidth in units of GB/s
	- $R_{B}$ : Number of bytes read per kernel
	- $W_{B}$ : Number of bytes written per kernel
	- $t$ : elapsed time given in seconds
	- Example code 在 saxpy 中
		- 裡面的 N\*4 代表從 word 轉成 byte
		- \*3 代表 3 次 memory movement
## Measuring Computational Throughput
- $GFLOP/s_{Effective} = 2N / (t * 109)$
- 這裡的 2 代表的是 multipy-add operation is measured as two FLOPs
## Summary
- 可以用 `cudaEvent` 來記錄 CUDA API call 的 event time
- 以下是我跑出來的結果, 不清楚為什麼每次跑出來的 performance 其實差蠻大的 ![[Screenshot 2024-12-05 at 2.30.33 PM.png]]
- 但我想說後來都用 nsys, 好像糾結這個的實際意義就相對比較小了