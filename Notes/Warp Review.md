#GPU #Architecture #Warp 
## What is Warp
- Warp 是 GPU 執行的最小單位 (一次就是執行這32個 thread)
- 一個 warp 中有 32 個 thread
- Block 與 warp 的 relation
	- 在多數的 device 上, 一個 block 會有 maximum  1024 個 thread
	- 一個 block 中, warp 的切割是從 x-axis 開始切 → consider 32\*32 block, the cutting is :
		- (0, 0) → (31, 0) : warp 0
		- (0, 1) → (31, 1)  : warp 1
		- ...
		- (0, 31) → (31,31) : warp 31
## Warp 的重要性
- Warp 是 GPU 執行的最小單位 (物理上的最小單位)

## Warp  跟 SM 的關係
- 用 V100 舉例 :
	- 1 block 最多可以有 1024 個 thread (多數 GPU up-limit 都是 1024)
	- V100 中有 80 個 SM
	- 每個 SM 有 64 warp, 每個 warp 32 thread → 每個 SM 有 64 * 32 = 2048 threads
	- 總共有 2048 * 80 threads
	- Warp 由 warp scheduler 管理, 在 V100 中一個 SM 有 4 個 warp scheduler
	- 每個 cycle warp scheduler 可以 dispatch 1 個 instruction
## Transaction size
- 代表 memory coalescing 的大小
- 以 V100 而言：
	- global/local mem, L2-cache, HBM 是 32 bytes 
	- shared mem 是 128 bytes (但是 shared memory 沒有 memory coalescing)

## Memory Coalescing (Global mem.)
- GPU 可以把一個 warp 中多個 memory access 結合, 減少 memory access 的次數
- 以 V100 舉例 : 
	- For load :
		- 128 bytes for 32-bit data types (int, float)
		- 64 bytes for 64-bit data types (double)
	- For store
		- 128 bytes for all data types
	- 如果 access 的大小在上述的大小內, 那就可以在一次 memory access 中完成
	- 如果超過, 就要增加 memory access 的數量, end-to-end time 就會增加
## Memory Banking && Bank Conflict (Shared mem.)
- GPU 的 shared memory 是以 bank 的架構設計, 如果同個 warp 中的 thread 同時 access 同一個 bank 的不同 address, 就會變成 serialize operation (如果是 access 相同的 address 則不會有影響)
- Solution : 
	- Memory padding 
	- Address linearization
- 以 V100 舉例：
	- 每個 bank 的 bandwidth 是 32bit (4 bytes) per clock cycle
	- V100 的每個 shared memory 有 32 個 bank
	-  所以如果 access 了同樣的 bank, 就會導致 performance drop

## SM 與 Block 的關係
- 1 個 Block 只能跑在 1 個 SM 上, 所以 Block 的 Register 跟 Share memory 才會被限制在 SM 的 spec 上
- 這裡要備註 : 多個 block 可以跑在 1 個 SM 上
- 但是在 GPU 上一個 block 可以跑的 thread 上限跟 一個 SM 可以跑的 thread 上限可能不一樣, 這點要特別注意 
	- e.g. 在 p100 上, 一個 block 可以跑 1024 個 thread, 而一個 SM 可以跑 2048 個 thread, 所以一個 SM 最少可以跑 2 個 block
- <font color="#ff0000">一個 block 用一個 shared memory </font>
