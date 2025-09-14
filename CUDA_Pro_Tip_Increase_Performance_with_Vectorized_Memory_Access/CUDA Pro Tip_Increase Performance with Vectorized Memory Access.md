#CUDA #Coding #Memory 
Ref : https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

## Outline
- 這篇主要介紹了 GPU 的 vectorization, 使用 vectorization 理論上可以增加 memory throughput, 但是在實驗後我發現 throughput 實際上下降了(或是沒上升, 跟上升不如預期), 所以可能還需要與人討論才能 performance 好不好的結論 → 與 Jerry 討論過了
## Original Code
- 這篇文章主要用 array copy 作為舉例, example code 裡有定義 `MAX_BLOCKS`, 我分別測試了 `MAX_BLOCKS` 是 65535 與 `MAX_BLOCKS` 無上限的兩種 code.
- 以下是 scalar solution 的 bandwidth (`MAX_BLOCKS == 65535` ) : ![[kernel_Copy_bandwidth_with_MAX_BLOCKS.png]]
- 以下是 scalar solution 的 bandwidth (`MAX_BLOCKS`無上限)![[kernel_copy_bandwidth_VS_array_size.png]]
- 可以看到在 500 GB/s 左右會形成一個平原, 我認為這部分應該可以用 roofline model 來解釋 #TODO 
- 其實到這之前都跟文章的數據差不多, assembly code 也跟預期的一樣 : 
	``` python
	/*0088*/                   IADD.X R3, R7, c[0x0][0x144] ;               
	/*0090*/                   LDG.E R2, [R2] ;                             
	/*0098*/                   MOV R6, c[0x0][0x8] ;                        
	
	...
                                                                        
	/*00e8*/                   STG.E [R4], R2 ;                             
	```

## Vectorized Code
- 接下來就是按照文章的方法把 code 改成 vectorized format, 但是從這就發現 performance 與預期的不相同 : 
	- 在文章中作者做出來的 performance 其實就沒有很理想, 畢竟 vectorize 後理論上 performance 要提升至少一倍 (從平程的經驗來看), 但是作者的圖表幾乎只提升了 10% (還是在最好的 performance 下) %![[Pasted image 20241228152427.png]]
	- 但當我自己做實驗時發現 : 我的實驗結果甚至連 10% performance 都沒辦法取得, 甚至在 `MAX_BLOCKS`無上限時, performance 會嚴重落後 scalar : 
		- `MAX_BLOCKS`無上限 : ![[Bandwidth_Comparison_P100_unlimited.png]]
		- `MAX_BLOCKS == 65535` :![[Bandwidth_Comparison_P100_Max_Blocks.png]]
	- 可看到就算在限制 `MAX_BLOCKS`時, 整體 performance 也只是跟 scalar 比起來時好時壞, 並沒有明顯的 out perform, 但我並不清楚為什麼會有這個現象, 畢竟這份 code 只有單純做 data transfer, 理論上要有更好的 performance. 
	- 我有懷疑過是否是程式碼寫錯了, 但是用 nsight 去看實際的運行時間確實沒有錯, 所以目前還找不到詳細原因
- 可以看到 assembly code 也跟文章提到的相同 : 
	- vector2
	``` python
	/*00d8*/                   LDG.E.64 R2, [R2] ; 
	/*0130*/                   STG.E.64 [R4], R2 ; 
	```
	-  vector4
	``` python
	/*00f0*/                   LDG.E.128 R4, [R4] ;
	/*0148*/                   STG.E.128 [R2], R4 ;  
	```

## Conclusion
- 目前得到的結論是 vectorize 的 performance 不一定會提升, 而且無法確定為何不會提升, 所以目前不會使用 vectorization, 直到找到 performance drop 的原因.
## Takeaway 
- 文章有提到的 ` %> cuobjdump -sass executable` 蠻好用的, 可以看到 assembly code.
- 文章有提到 array 需要 align data type 才能夠使用 vectorize, 這點需要注意. 
	- `reinterpret_cast<int2*>(d_in+1)` → invalid, 因為 `d_in+1`不能整除 `sizeof(int2)`
	- `reinterpret_cast<int2*>(d_in+2)` → valid
- 這裡學到了 P100-16GB 的 max memory bandwidth 是 700 GB/s
- 還有學到 max dimension of block is : 
	- X dimension: 2,147,483,647 (2^31 - 1)
	- Y dimension: 65,535
	- Z dimension: 65,535
- 所以可以設定 max block == 65535

## 與 Jerry 討論後的想法
- 可能的原因 : 從文中看起來這裡的 vectorization 並不是硬體加速, 只是類似 Pragma unroll 的行為, 所以才不會有 performance 的提升.

## 跑在 H100 後的效果 
![[Bandwidth_Comparison_H100.png]]

## 與 Reese 討論的結果 
- 可能要清空 L2 cache
- edge case detection 寫在 host, device 裡面不要寫
- P100 performance 不好可能的原因是 : 新的 cuda 對 P100 有 bug ![[Pasted image 20250115155458.png]]
- 這是跟Reese 討論後得到 performance 只提升 10% 的原因(與 Jerry 討論的結論大同小異) :  ![[Pasted image 20250115155552.png]]
- 學到了 ncu command `ncu -f --clock-control none --set=full -o <ncu_file_name> ./exe`
- coding style 調整 
	- kernel code 裡面要乾淨, 只有確定要 vector 的才 call kernel code, 不要把 edge case 判斷放在 kernel code 中
