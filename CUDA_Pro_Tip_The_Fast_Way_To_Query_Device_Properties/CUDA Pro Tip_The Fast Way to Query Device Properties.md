#Coding #CUDA #Debug #Profiling 
Ref : https://developer.nvidia.com/blog/cuda-pro-tip-the-fast-way-to-query-device-properties/

## Just the Facts You Need
- `cudaGetDeviceProperties()` : 會給所有的 properties, 代價是很慢
- `cudaDeviceGetAttribute` : 只會給一個, 但是很快

## Benchmarking Device Attribute Queries
- 用 P100 跑的結果 : ![[Screenshot 2024-12-09 at 8.20.58 PM.png]]
- 大概差了 900 倍, 如果是寫死在 code 裡更要注意

## Caution: Some Attributes are Expensive
- 有些 attribute 就算用 `cudaDeviceGetAttribute` 還是很慢 : 
	- `cudaDevAttrClockRate`
	- `cudaDevAttrKernelExecTimeout`
	- `cudaDevAttrMemoryClockRate`
	- `cudaDevAttrSingleToDoublePrecisionPerfRatio`.
	

## 備註
- Example code 裡的 macro 蠻讚的, 可以拿來用 : 
``` python
#define CUDA_CHECK(call)                                  \
	do {                                                  \
		cudaError_t status = call;                        \
		if(status != cudaSuccess) {                       \
			printf("FAIL: call='%s'. Reason:%s\n", #call, \
				cudaGetErrorString(status));              \
		return -1;                                        \
	}                                                     \
} while (0)
```