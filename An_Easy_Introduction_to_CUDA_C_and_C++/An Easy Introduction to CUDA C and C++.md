#CUDA #Unified_Memory 
Ref : https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/

## Evaluating performance diff of ordinary memory movement and unified memory
- 這篇與 An even easier introduction to CUDA 大致上相近, 主要是探討 unified memory 與 normal memory management 的 performance 差異
	- saxpy : 235
	- saxpy_unify : 260 ~ 280 (average 270) && First time will be long (310)
	- 從數據上來看, 用 cudaMalloc 的 solution performance 大約會比 unify 的 solution 好 10%-15%
- 另外也做了把 init 搬移到 GPU 的版本, 
	- saxpy : 240
	- saxpy_unify : 250 左右
	- 結論而言 performance 仍然是 cudaMalloc > unify memory, performance diff 會下降

## Conclusion 
- cudaMalloc is generally better , 但是如果在 random access的情況下, 可能就是 unified memory 會比較好, 因為他是 demand paging

今天做了測試後發現 performance 與預期不同, 想把資料整理後討論 #TODO 