#CUDA #Debug 
Ref : https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/
## Querying Device Properties
- 這裡討論了如何用 cudaDeviceProp 來 access device spec
	- `cudaGetDeviceCount()`
## Compute Capability
- Compute Capability 代表了跟 Tesla C870 performance  的比值
- 看了一下 P100 的 compute capability 是 6.0
- 以下是 P100-16G 的 spec![[Screenshot 2024-12-07 at 11.10.24 AM.png]]
	- 這裡的筆記我放在 [[Warp Review#SM 與 Block 的關係]]
- 這邊也提到了可以用 `-arch=sm_xx` 來調整 compute capability
## Handing CUDA Errors
- 這邊提到如何在 runtime 時 detect CUDA Error, 我的理解是會先用這裡的方法來 detect 後再用 compute-sanitizer 來做詳細 detection. → code 在 `saxpy_error_detection.cu`
- 比較重要的應該是有分 `cudaDeviceSynchronize()` 跟 `cudaGetLastError()`
	- **`cudaDeviceSynchronize()`** : blocking , 所以可以 detect 所有在這行之前的 error
	- **`cudaGetLastError()`** : non-blocking, 所以可能 detect 不到這行之前的 error (我覺得比較難用, 主要是避免 performance 掉, 所以要更注意放置的地方), 另外也可以考慮用 `cudaPeekAtlastError()`, 這就等需要時再考慮要用哪個
	