#Unified_Memory #Coding #Memory 
Ref : https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/

> The upside is that if you have a lot of compute in your kernel then the migrations can be amortized or overlapped with other computation, and in some scenarios Unified Memory performance may even be better than a non-overlapping `cudaMemcpy` and kernel approach


- 跑 100 次的前提下, 三個 solution 的 performance 幾乎一樣![[Pasted image 20250102132319.png]]
	- 這個結果與 blog 提到的結果並不相同, 最明顯的差異是 on-demand paging 的 performance 有顯著提升 (與另外兩個幾乎一樣的 performance), 算是一個有趣的現象, 等稍後可以用國網中心的 gpu (V100 with NVLink) 後會再跑看看, 來觀測 performance 差異 #TODO 
- 但是這裡 performance 會一樣是因為我下了 `cudaDeviceSynchronize();`, 我相信如果有 overlap, 整體 performance 會有差別, 其實可以測看看 end-to-end time, 來測試到底哪份 code 的 performance 比較好. 
	- 我的猜測是 Async > PrefetchAsync > on-Demand Paging
	- 實驗結果 : ![[Pasted image 20250102135424.png]]
	- 發現了有趣的現象 : pre-allocating 的 performance 很好, 其餘兩個很差, 算是出乎意料, 
	- 可以得到結論 : 
		1. <font color="#ffc000">pinned memory + async</font> 的效益是最大的
		2. prefetch 如果放得不好, 可能會導致 performance 不增反減
		3. on-demand paging 的優化做得好到已經跟 prefetching 幾乎相同了. (在這種簡單 case 下)

## Warp-Per-Page Approach
