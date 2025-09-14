#Memory #Unified_Memory 
Ref : https://developer.nvidia.com/blog/unified-memory-cuda-beginners/

## Fast GPU, Fast Memory...Right ?
- 這段探討了為什麼 P100 上面跑  add_gird.cu 時的 performance 會與跑 add_block 時的 performance 接近 (都是 3ms)
	- 當 K80 GPU 在跑時, 因為 pre-Pascal GPU 沒有 virtual page fault 機制, 所以需要將所有會 access 到的 page 都先搬移到 GPU 中 (就算是 unified memory 也一樣), 但是這個搬移的過程並不會被算在 add() function 的計算時間中
	- 但是當 P100 GPU 在跑時, 因為 virtual page fault, 所以只有在那個 page 被 access 時, 才會開始 migration. 這樣的好處是可以避免 locality 以及 excessive page faults, 但是這段搬移時間就會被算入到 add() function 運算時間中.
- 結論是 page migration overhead 有沒有被觀測到

## What Should I Do About This ?
概念上 memory migration 的時間是無法避免的, 這裡只是提到要如何單純展現 add() 的計算時間, 而不會讓 computation time 與 data movement time 被合在一起被觀測
1. Initialize the Data in a Kernel
	- 如果把 initialization 移到 GPU 中, 這樣一開始 page 的初始化就會在 GPU memory 中, 所以便不會有 page fault 產生 <font color="#c00000">(這個方法可以避免掉 page migration overhead)</font>
	- Code 在 add_grid_init.cu 中
	- Profiling : ![[Screenshot 2024-12-02 at 1.37.31 PM.png]]
		- 可以看到 total page faults 從 36 下降到 12 (這裡的 12 page fault 按照作者說是因為 device-to-host 導致)
		- add() 的運算下降到 29 us
	- 從 profiling 中可以觀測到一個現象 : 為什麼 init() 的 overhead 這麼高 ?
		- 觀測到一個現象 : init() 與 add() code structure 基本上相同, 但是兩者的 overhead 卻相差非常大 <font color="#c00000">20ms vs. 30us</font>
		- 經過實驗後可以發現 overhead 高是因為第一次 trigger cuda function 時, 因為 cuda 使用的是 dynamic linking library, 所以需要 load library, 這段時間會被算在 init 的 runtime 中 (如下圖)![[Screenshot 2024-12-02 at 2.03.17 PM.png]]
		- 後續寫了一個跑了100次 init() 的實驗, 即可發現這個現象便不會發生 → code 在 `add_grid_many_init.cu`
		- 也可以從 profiling 中看到 init() 的 avg. time 是 77us, 與當時在台積觀測到的現象相同![[Screenshot 2024-12-02 at 2.06.14 PM.png]]
2. Run It Many Time
	- 這個方法就是跑 100 次 add, 觀測 avg. time → code 在 `add_grid_many_init.cu`
	- 因為只有在第一次 run 會觸發 page fault, 所以 avg. runtime 其實不會高
	- Profiling : ![[Screenshot 2024-12-02 at 2.10.39 PM.png]]
		- 可以看到 add() 的 Max = 3ms, min = 28us, avg. = 59us, 也與理論相符
3. Prefetching
	- 這個方法是先用 prefetch function 來將資料先搬到 GPU 中, 這樣就不會有 add() 的 migration overhead (我覺得跟最早的 memory 搬法差不多) → code 在 `add_grid_prefetch.cu`
	- Profiling : ![[Screenshot 2024-12-02 at 2.21.49 PM.png]]
		- 可以看到 add() 的 overhead 是 30us, 與理論相符
	- 這裡順便把 block 版本也寫了 prefetch 版本, Profiling 如下 : ![[Screenshot 2024-12-02 at 2.24.04 PM.png]]
		- 可以看到 add 的 overhead 其實差異蠻大的 1.8ms vs. 30us
		- 可以得知 grid solution performance 好很多
## A Note on Concurrency
- 這段提到在 cuda 6.0 後的 unified memory 中 , CPU 與 GPU 可以同時 access memory, 所以會導致 race condition
- 這邊的簡單 example 中是用 cudaDeviceSynchronize() 來解這個問題, 但也可以用 atomic operation 來解決
## The Benefits of Unified Memory on Pascal and Later GPUs
- Pascal之後的 virtual memory 增加到49-bit virtual addresses, 可以避免 OOM 產生 (但是為什麼現在還是有 OOM 發生 ?) #TODO #NeedToAsk
- 另外提到 Pascal 後的 data center gpu 有 system-wide atomic memory operation, 可以跨卡內的 atomic, 在 multi-gpu 中很有用
- 最後則是 demand paging 可以降低 overhead (避免多 load 很多用不到的 data 進來 gpu)
- 結論, Unified memory 在 Pascal 後的 GPU 有 : 
	1. 49-bit virtual address can avoid OOM
	2. System wide atomic operation, useful in multi-gpu programming
	3. Demand paging to reduce loading page overhead
## Question
- Does unified memory will give CUDA programming better performance ? or only easier for coding but bad performance ? 
	- 這裡做了測試實驗 : [[An Easy Introduction to CUDA C and C++]], 結論是 unified memory performance 會掉, 但是如果今天不會 access 到所有搬入的 memory, 那 unified memory 可能會比較好, 因為他是 demand paging
