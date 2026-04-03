// matmul_kernel.cu
// Compute C = alpha * A * B + beta * C
#include <cuda_runtime.h>

// tweak this tile size as you optimize
#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif


extern "C"
__global__ void kernel1_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
) {
    // !! 會忘記
    const int cRow = blockIdx.x * blockDim.x + threadIdx.x; // for A(M,K) 
    const int cCol = blockIdx.y * blockDim.y + threadIdx.y; // for B(K,N)

    if (cRow < M && cCol < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; i++) {
            tmp += A[cRow * K + i] * B[i * N + cCol];
        }
        C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
    }
}


extern "C"
__global__ void kernel2_coalescing(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
) {
    int BLOCKSIZE = 32;
    const int cRow = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE; // for A(M,K)
    const int cCol = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE; // for B(K,N)

    if (cRow < M && cCol < N){
        float tmp = 0;
        for (int i = 0; i < K; i++) {
            tmp += A[cRow * K + i] * B[i * K + cCol];
        }
        C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
    }
}

extern "C"
__global__ void kernel3_shared(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
) {
    const int BLOCKSIZE = 32;
    
    const int threadRow = threadIdx.x / BLOCKSIZE; 
    const int threadCol = threadIdx.x % BLOCKSIZE;
    const int blockRow = blockIdx.x;
    const int blockCol = blockIdx.y;
    
    __shared__ float As[BLOCKSIZE + 1][BLOCKSIZE + 1];
    __shared__ float Bs[BLOCKSIZE + 1][BLOCKSIZE + 1];
    
    if (blockRow * BLOCKSIZE + threadRow < M && blockCol * BLOCKSIZE + threadCol < N) {
        
        A += blockRow * BLOCKSIZE * K; // Row = cRow, Col = 0
        B += blockCol * BLOCKSIZE; // Row = 0, col = cCol
        C += blockRow * BLOCKSIZE * N + blockCol * BLOCKSIZE; // Row = cRow, Col = cCol

        float tmp = 0;

        for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
            As[threadRow][threadCol] = A[threadRow * K + threadCol];
            Bs[threadRow][threadCol] = B[threadRow * N + threadCol];

            __syncthreads();
            
            A += BLOCKSIZE;
            B += BLOCKSIZE * N;
            
            for (int i = 0; i < BLOCKSIZE; i++) {
                tmp += As[threadRow][i] * Bs[i][threadCol];
            }

            __syncthreads();
        }
        C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
    }


}

extern "C"
__global__ void kernel4_1DBlock(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
) {
    const int BM = 64; // size of each block responsible
    const int BN = 64; // size of each block responsible
    const int BK = 8;
    const int TM = 8; // len of each thread responsible
    
    // If we flip x and y here we get ~30% less performance for large matrices.
    // The current, 30% faster configuration ensures that blocks with sequential
    // block IDs access columns of B sequentially, while sharing the same row of A.
    // The slower configuration would share columns of A, but access into B would
    // be non-sequential. So the faster configuration has better spatial locality
    const int blockRow = blockIdx.y; // !!
    const int blockCol = blockIdx.x; // !!
    const int threadRow = threadIdx.x / BN;
    const int threadCol = threadIdx.x % BN;

    const int AsRow = threadIdx.x / BK;
    const int AsCol = threadIdx.x % BK;
    const int BsRow = threadIdx.x / BN;
    const int BsCol = threadIdx.x % BN;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    if (blockRow * BM + threadRow < M && blockCol * BN + threadCol < N) {
        A += blockRow * K * BM;
        B += blockCol * BN;
        C += blockRow * N * BM + blockCol * BN;

        // shared memory pushing
        float TMP[TM] = {0.0};
        for (int i = 0; i < K; i+=BK) {
            As[AsRow][AsCol] = A[AsRow*K + AsCol];
            Bs[BsRow][BsCol] = B[BsRow*N + BsCol];

            __syncthreads();

            A += BK;
            B += BK * N;

            // 1D-Blocktiling
            for (int j = 0; j < BK; j++) {
                float tmp = Bs[j][threadCol];
                for (int k = 0; k < TM; k++) {
                    TMP[k] += As[threadRow * TM + k][j] * tmp; // !! 易忘
                }
            }
                    
            __syncthreads();
        }

        
        for (int i = 0; i < TM; i++) {
            // !! often make mistake
            C[(threadRow * TM + i) * N + threadCol] = alpha * TMP[i] + beta * C[(threadRow * TM + i) * N + threadCol];
        }
    }
}

extern "C"
__global__ void kernel5_2DBlock(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
) {
    const int BM = 128;
    const int BN = 128; 
    const int BK = 16;
    const int TM = 8; 
    const int TN = 8; 

    const int numThreadsPerBlock = BM * BN / (TM * TN);

    assert(numThreadsPerBlock == blockDim.x);

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    const int threadRow = threadIdx.x / (BN/TN); // !! 易忘
    const int threadCol = threadIdx.x % (BN/TN); // !! 易忘

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    A += blockRow * BM * K;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    const int AsRow = threadIdx.x / BK;
    const int AsCol = threadIdx.x % BK;
    const int strideA = numThreadsPerBlock / BK; 
    // !! As 中的一個 col 要用多少 thread
    // 解釋 : 一個 As[BM][BK], 共有 BK 個 column, 
    // 所以 numThreadsPerBlock / BK 代表 1 個 column 會被分配到多少個 thread, 
    // 舉例來說 : As = 128 * 8, numThreadsPerBlock = 256, 所以 256 / 8 = 32, 
    // 也就是每個 column 會需要 32 個 thead 負責搬移, 這樣反而有 memory coalescing
    
    const int BsRow = threadIdx.x / BN;
    const int BsCol = threadIdx.x % BN;
    const int strideB = numThreadsPerBlock / BN;

    float TMP[TM][TN] = {0};

    float regM[TM] = {0};
    float regN[TN] = {0};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // !! 會忘記 Row + loadOffset
        for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) { 
            As[AsRow + loadOffset][AsCol] = A[(AsRow + loadOffset) * K + AsCol]; 
        }
        for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) { // !! < BK 有寫錯過
            Bs[BsRow + loadOffset][BsCol] = B[(BsRow + loadOffset) * N + BsCol];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int dotIdx = 0; dotIdx < BK; dotIdx++) { //!!
            for (int i = 0; i < TM; i++) {
                regM[i] = As[threadRow * TM + i][dotIdx]; // !! 會忘記 threadRow * TM
            }
            for (int i = 0; i < TN; i++) {
                regN[i] = Bs[dotIdx][threadCol * TN + i]; // !! 會忘記 threadCol * TN
            }
            for (int idxM = 0; idxM < TM; idxM++) {
                for (int idxN = 0; idxN < TN; idxN++) {
                    TMP[idxM][idxN] += regM[idxM] * regN[idxN];
                }
            }
        }
        __syncthreads();
    }
    // write result back
    for (int idxM = 0; idxM < TM; idxM++) {
        for (int idxN = 0; idxN < TN; idxN++) {
            C[(threadRow * TM + idxM) * N + threadCol * TN + idxN] = 
                alpha * TMP[idxM][idxN] + 
                beta * C[(threadRow * TM + idxM) * N + threadCol * TN + idxN];
        }
    }
}

extern "C"
__global__ void kernel6_Vectorization(
    float* A,
    float* B,
    float* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
) {
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8; 
    const int TN = 8;

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadRow = threadIdx.x / (BN / TN);
    const int threadCol = threadIdx.x % (BN / TN);

    __shared__ float As[BK][BM]; // !! 用 transpose 存入 share memory 中, 避免 bank conflict
    __shared__ float Bs[BK][BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const int AsRow = threadIdx.x / (BK / 4);
    const int AsCol = threadIdx.x % (BK / 4);
    const int BsRow = threadIdx.x / (BN / 4);
    const int BsCol = threadIdx.x % (BN / 4);

    float TMP[TM][TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        float4 tmp = reinterpret_cast<float4 *>(&A[AsRow * K + AsCol * 4])[0];
        As[(AsCol*4 + 0)][AsRow] = tmp.x;
        As[(AsCol*4 + 1)][AsRow] = tmp.y;
        As[(AsCol*4 + 2)][AsRow] = tmp.z;
        As[(AsCol*4 + 3)][AsRow] = tmp.w;

        reinterpret_cast<float4 *>(&Bs[BsRow][BsCol*4])[0] = 
            reinterpret_cast<float4 *>(&B[BsRow * N + BsCol * 4])[0];

        __syncthreads();

        A += BK;
        B += BK * N;

        for(int dotIdx = 0; dotIdx < BK; dotIdx++) {
            for (int i = 0; i < TM; i++) {
                regM[i] = As[dotIdx][threadRow * TM + i];
            }
            for (int i = 0; i < TN; i++) {
                regN[i] = Bs[dotIdx][threadCol * TN + i];
            }
            for (int idxM = 0; idxM < TM; idxM++) {
                for (int idxN = 0; idxN < TN; idxN++) {
                    TMP[idxM][idxN] += regM[idxM] * regN[idxN];
                }
            }
        }
        __syncthreads();

    }
    for (int idxM = 0; idxM < TM; idxM++) {
        for (int idxN = 0; idxN < TN; idxN += 4) {
            float4 tmp = reinterpret_cast<float4 *>(
                &C[(threadRow * TM + idxM) * N + threadCol * TN + idxN])[0];
            tmp.x = alpha * TMP[idxM][idxN] + beta * tmp.x;
            tmp.y = alpha * TMP[idxM][idxN+1] + beta * tmp.y;
            tmp.z = alpha * TMP[idxM][idxN+2] + beta * tmp.z;
            tmp.w = alpha * TMP[idxM][idxN+3] + beta * tmp.w;

            reinterpret_cast<float4 *>(
                &C[(threadRow * TM + idxM) * N + threadCol * TN + idxN])[0] = tmp;
        }
    }
}

namespace wt {
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                             float *As, float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    // float4 tmp;
    // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
    //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
    // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
    //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // populate registers for whole warptile
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // execute warptile matmul
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // calculate per-thread results
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

} // namespace wt


extern "C"
__global__ void kernel10_Warptiling(
    float* A,
    float* B,
    float* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
) {
    const int NUM_THREADS = 128;
    const int WARPSIZE = 32;
    const int BM = 128;
    const int BN = 128;
    const int BK = 16;
    const int WM = 64;
    const int WN = 64;
    const int WNITER = 4;
    const int TM = 8;
    const int TN = 4;
    const int NUM_WARPS = NUM_THREADS / 32;
    
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int warpIdx = threadIdx.x / WARPSIZE;
    const int warpCol = warpIdx % (BN / WN);
    const int warpRow = warpIdx / (BN / WN);

    const int WMITER = (WM * WN) / (WARPSIZE * TM * TN *WNITER);
    const int WSUBM = WM / WMITER;
    const int WSUBN = WN / WNITER;

    const int threadIdxInWarp = threadIdx.x % WARPSIZE;
    const int threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const int threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;

    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int rowStrideA = (NUM_THREADS * 4) / BK;
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);
    const int rowStrideB = NUM_THREADS / (BN / 4);

    float threadResults[WMITER * TM * WNITER * TN] = {0};
    float regM[WMITER * TM] = {0};
    float regN[WNITER * TN] = {0};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
        __syncthreads();
        wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol, threadRowInWarp, threadColInWarp);

        A += BK; 
        B += BK * N;
        __syncthreads();
    }

    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++) {
        for (int wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++) {
            float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for (int resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    float4 tmp = reinterpret_cast<float4 *>(&C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN])[0];
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN;

                    tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
                    tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
                    tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
                    tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;

                    reinterpret_cast<float4 *>(&C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}

// const int tile = 32;
// gridDim3(int((M+tile-1)/tile), int((N+tile-1)/tile));
// blockDim3(tile * tile, );

extern "C"
__global__ void kernel_test(
    float* A,
    float* B,
    float* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
){
    const int BM = 128;
    const int BN = 128;
    const int BK = 16;
    const int TM = 8;
    const int TN = 8;


    const int numThreadsPerBlock = BM * BN / (TM*TN);
    // const int BLOCKSIZE = 32;
    const int blockRow = blockIdx.x;
    const int blockCol = blockIdx.y;
    const int threadRow = threadIdx.x/(BN/TN);
    const int threadCol = threadIdx.x%(BN/TN);

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int AsRow = threadIdx.x / BK;
    const int AsCol = threadIdx.x % BK;
    const int AsStride = numThreadsPerBlock / BK;
    const int BsRow = threadIdx.x / BN;
    const int BsCol = threadIdx.x % BN;
    const int BsStride = numThreadsPerBlock / BN;

    float TMP[TM][TN] = {0};
    float regM[TM] ={0};
    float regN[TN] ={0};

    A += blockRow * BM * K;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    if (blockRow * BM + threadRow * TM < M && blockCol * BN + threadCol * TN < N) {
        for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
            for (int i = 0; i < BM; i+=AsStride){
                As[AsRow + i][AsCol] = A[(AsRow + i) * K + AsCol];
            }
            for (int i = 0; i < BK; i+=BsStride){
                Bs[BsRow + i][BsCol] = B[(BsRow +i) * N + BsCol];
            }
            
            __syncthreads();

            A += BK;
            B += BK * N;
            
            for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
                for (int i = 0; i < TM; i++) {
                    regM[i] = As[threadRow * TM + i][dotIdx];
                }
                for (int i = 0; i < TN; i++) {
                    regN[i] = Bs[dotIdx][threadCol * TN + i];
                }
                for (int i = 0; i < TM; i++) {
                    for (int j = 0; j < TN; j++){
                        TMP[i][j] += regM[i] * regN[j];
                    }
                }
            }

            __syncthreads();
        }
        for (int i = 0; i < TM; i++) {
            for (int j = 0; j < TN; j++){
                C[(threadRow * TM + i) * N + threadCol * TN + j] = alpha * TMP[i][j] + beta * C[(threadRow * TM + i) * N + threadCol * TN + j];
            }
        }
    }
}