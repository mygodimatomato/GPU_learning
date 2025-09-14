#CUDA #Coding #Memory 
Ref : https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/

## Introduction
作者在這段提到了幾個 general guideline for host-device data transfer
- 盡量減少 data transfer from host ↔ device
- Pinned memory 可以提昇 data transfer performance
- Batching small transfer 可以減少 transfer overhead
- Overlap transfer with other operation → 這段放在 [[How to Overlap Data Transfer in CUDA C&&C++]]
## Measuring Data Transfer Times with `nvprof`
- 這段先提到了如何用 nvprof 來測量 cuda 的傳輸時間
- 以下是我跑出來的結果 ![[Screenshot 2024-12-15 at 4.13.11 PM.png]]
	- 可以看到 Device to Host 時間比較長, 蠻好奇原因是什麼, 所以對那塊 memory 做了額外操作, 但是還是一樣的數據, 也嘗試做了 nsys, 但是結果也一樣, 看不出發生的原因![[Pasted image 20241215162031.png]]![[Pasted image 20241215163537.png]]
	- 可以從後續的 bandwidth 測試發現 HtoD 跟 DtoH 的比例並沒有差異這麼大, 所以很困惑為什麼會有這麼明顯的 transfer time 差異, **目前想不出可以解釋的原因**. ![[Pasted image 20241215164225.png]]
	- **當我跑了 1000 次後, 其實整體時間會非常接近(接近 bandwidth 測試出來的結果),** 目前猜測是 system initialization 的原因(但是 nsys 沒有看到背後有東西在跑). ![[Pasted image 20241215164124.png]]
	- 目前想到可能原因是 pinned memory setup, 但是還在想要如何 validate #TODO 
## Minimizing Data Transfers
這段強調 GPU transfer time 是目前的 bottleneck, 所以需要優先考慮, 而不是考慮 execution time. 甚至有時候雖然 GPU 跑比較慢, 也要把 computing 留在 GPU 上, 以減少 data transfer time.

## Pinned Host Memory
下面這張圖解釋了 Pageable Data Transfer vs. Pinned Data Transfer ![[Pasted image 20241215164726.png]]
- Default memory transfer 是 pageable, 但是就需要經過兩手才能完成 memory transfer. 但是如果用 Pinned memory 就只需要經過一手<font color="#f79646">(這裡的 default 並不是 unified memory, 而是一般的 cudaMemcpy)</font>
- **這裡需要釐清一件事 : pinned 跟 pageable data transfer 都是用 `cudaMemcpy()`, 差異是 pinned data transfer 需要做額外的 initialization(`cudaMallocHost()`)**
- Pinned memory usage : 
	- `cudaMallocHost()` : Allocates pinned memory on the host.
	- `cudaHostAlloc()` : Allocates pinned memory on the host with more advanced control via **flags**.
		- 這個可以讓 multi-gpu 同時 access 到這塊 memory
	- `cudaFreeHost()` : deallocates
- 以下是 Titan profile 出來的結果 :![[Pasted image 20241215172440.png]]
- 這裡有提到 over-allocate pinned memory 會影響到 OS 的運作
## Batching Small Transfers
這段的重點是如果有多個 array 要搬進 gpu, 可以把它們放在同一個 array 中 (用 2D array 來表示), 並用 `cudaMemcpy2D()`, `cudaMemcpy3D()` 來將這兩個 array同時搬進去, 這樣就能避免掉多次呼叫 cudaMemcpy 的 overhead.