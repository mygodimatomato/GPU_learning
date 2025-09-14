// matmul_kernel.cu
// Compute C = alpha * A * B + beta * C
#include <cuda_runtime.h>

// tweak this tile size as you optimize
#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

extern "C"
__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* C,
    int N,
    float alpha,
    float beta
) {
    // global row/col index for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // guard against threads outside the N×N matrix
    if (row >= N || col >= N) return;

    // accumulator for the dot product
    float sum = 0.0f;

    // === Naïve version ===
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }

    // writeback with alpha/beta scaling
    C[row * N + col] = alpha * sum + beta * C[row * N + col];

    /* === Tiled version stub (optional next step) ===
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    sum = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        int tiledRow = t * TILE_SIZE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < N && tiledCol < N) ? A[row * N + tiledCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (tiledRow < N && col < N) ? B[tiledRow * N + col] : 0.0f;
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
    */
}
