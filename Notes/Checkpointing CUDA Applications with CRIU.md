#GPU #checkpoint_restart 
Ref : https://developer.nvidia.com/blog/checkpointing-cuda-applications-with-criu/
Repo : https://github.com/NVIDIA/cuda-checkpoint
## Intro 
- `cuda-checkpoint` : checkpoint and restore CUDA state within a Linux process
- combine `cuda-checkpoint` and `criu` to fully checkpoint CUDA application

## `cuda-checkpoint`
- Support display driver version 550 and higher
- Checkpoints && restore CUDA state of single linux process
- Suspend : 
	- Suspend process : 
		1. Any CUDA driver apis that launch work, manage resources, or otherwise impact GPU state are locked.
		2. Already submitted CUDA work, including stream callbacks, is completed.
		3. Device memory is coped to the host, into allocations managed by the CUDA driver.
		4. All CUDA gpu resources are released.
	- Only CUDA are paused, it didn't pause the CPU threads that are running the program
	- Threads can still call CUDA function (will block until CUDA is resumed)
	- Threads can access pinned memory (e.g. `cudaMallocHost`) (是 access pinned memory 的 host 端)
	- When a CUDA process is "suspended," it is **disconnected from the GPU hardware** at the operating system level. The GPU isn't actively linked to the process anymore. So CPU-based checkpointing tool like CRIU can save the process.
- Resume : 
	- Resume process : 
		1. GPU are re-acquired by the process.
		2. Device memory is copied back to the GPU and GPU memory mappings are restored at their original addresses.
		3. CUDA objects such as streams and contexts are restored.
		4. CUDA driver APIs are unlocked
## Example
- just see the blog, I may put something here if I think is important
## Functionality
- Still in development.
- x64 only.
- Only acts upon single process, not a process tree.
- Doesn't support UVM(Unified Virtual Memory) || IPC(Inter-Process Communication) memory. 
- Doesn't support GPU migration.
- Waits for already-submitted CUDA work to finish before completing a checkpoint.
- Doesn’t attempt to keep the process in a good state if an error (such as the presence of a UVM allocation) is encountered during checkpoint or restore. (如果在 checkpoint/restart 時掛掉, 就掛了, 不會做額外嘗試)