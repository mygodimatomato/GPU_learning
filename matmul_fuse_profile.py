import cupy as cp
import nvtx

# Define matrix size
N = 4096

# Define the fused elementwise operation
@cp.fuse()
def fused_alpha_beta(X, C, alpha, beta):
    return alpha * X + beta * C

@nvtx.annotate("Create matrices", color="orange")
def create_matrices(N):
    A = cp.random.randn(N, N, dtype=cp.float32)
    B = cp.random.randn(N, N, dtype=cp.float32)
    C = cp.random.randn(N, N, dtype=cp.float32)
    cp.cuda.Device(0).synchronize()
    return A, B, C

@nvtx.annotate("Fused computation", color="red")
def compute_fused(A, B, C, alpha, beta):
    X = cp.matmul(A, B)            # Matmul (cannot be fused)
    C = fused_alpha_beta(X, C, alpha, beta)  # Fused scaling and addition
    cp.cuda.Device(0).synchronize()
    return C

def main():
    # add a for loop to repeat the operation 10 times
    for i in range(10):
        print(f"Iteration {i+1}")
        A, B, C = create_matrices(N)
        alpha = 2.0
        beta = 0.5
        C = compute_fused(A, B, C, alpha, beta)
    # print("Fused Operation completed.")

if __name__ == "__main__":
    main()
