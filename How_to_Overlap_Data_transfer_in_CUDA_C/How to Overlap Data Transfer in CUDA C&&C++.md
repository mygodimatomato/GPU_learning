#CUDA #Coding #Memory #Stream 
Ref : https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
## CUDA Streams
概念上 stream 被分成兩種 : Default stream vs. Non-default streams
**Default stream(null stream)**
- Default stream 是 synchronizing stream, 這代表需要等 GPU 上其他 stream 完成 default stream 才會開始 (也可以理解成 default stream 的 op 會比其他 stream 先開始)
- CUDA 7 allow 每個 host thread 都有自己的 default stream #TODO 
- 作者提到可以這樣寫 :
``` c
  cudaMemcpy(d_a, a, numBytes, cudaMemcpyHostToDevice);
  increment<<<1,N>>>(d_a)
  myCpuFunction(b) // 把 cpu code 放在這可以避免 cpu stall
  cudaMemcpy(a, d_a, numBytes, cudaMemcpyDeviceToHost);
```

**Non-default streams**
- How to use : 
``` c
cudaStream_t stream1;
cudaError_t result;
result = cudaStreamCreate(&stream1)
result = cudaStreamDestroy(stream1)
```
- 然後用 `cudaMemcpyAsync` , `cudaMemcpy2Dsync()`, `cudaMemcpy3DAsync` 來做 data transfer

**Synchronization with streams**
- `cudaDeviceSynchronize()`: blocks the host until all previously issued op on device have completed. (overhead 非常大)
- `cudaStreamSynchronize(stream)`: only block until specific stream have done
- `cudaStreamQuery(stream)`: check without blocking the host.
## Overlapping Kernel Execution and Data Transfers
Several requirement for overlapping :
- 可以用 `deviceQuery` 檢查 GPU 有沒有 `deviceOverlap`的 prop.
- Overlapped execution && data transfer 都需要在 different non-default streams.
- 只有 pinned memory 才能 overlap

作者在此提到了兩個 steam 的寫法 : 
- V1 : 
``` c
for (int i = 0; i < nStreams; ++i) {
  int offset = i * streamSize;
  cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
  kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
  cudaMemcpyAsync(&a[offset], &d_a[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
}
```
- V2
```c
for (int i = 0; i < nStreams; ++i) {
  int offset = i * streamSize;
  cudaMemcpyAsync(&d_a[offset], &a[offset], 
                  streamBytes, cudaMemcpyHostToDevice, cudaMemcpyHostToDevice, stream[i]);
}

for (int i = 0; i < nStreams; ++i) {
  int offset = i * streamSize;
  kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
}

for (int i = 0; i < nStreams; ++i) {
  int offset = i * streamSize;
  cudaMemcpyAsync(&a[offset], &d_a[offset], 
                  streamBytes, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToHost, stream[i]);
}
```

這兩個寫法的主要差異如下圖 ![[Pasted image 20241217161905.png]]
- 可以看到早期的 GPU <font color="#f79646">只有一個 copy engine</font>, 所以兩種寫法的 performance 其實差異很大 (V1 幾乎沒有優化)

但是在偏近代的 GPU 中, 其優化表現則如下圖 : ![[Pasted image 20241217162125.png]]
- 在 C2050 中, 分別有兩個 copy engine (H2D engine, D2H engine), 所以可以平行化兩向的 data transfer, 因此 V1 的 performance 會有顯著提升
- 接著文章提問 : 為什麼 V2 的 performance 反而下降了 ?
- 作者解釋 : 
	- C2050 有 concurrent engine, 所以如果不同 stream 的 kernel back-to-back 被 trigger, 則會導致 scheduler 嘗試平行化這些 stream, 進而導致 scheduler 會在所有 kernel 都完成後才觸發 D2H Engine, 而不像 V1 是一完成就觸發, 進而導致 performance 下降. 
	- 這個現象在 compute capability 3.5 已經解掉了, 在 P100 中可以看到 V1 與 V2 有幾乎相同的 performance.![[Pasted image 20241219092532.png]]