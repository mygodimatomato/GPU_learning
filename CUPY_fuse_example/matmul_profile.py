import cupy as cp
import nvtx

# Define matrix size
N = 4096

# Create random matrices
A = cp.random.randn(N, N, dtype=cp.float32)
B = cp.random.randn(N, N, dtype=cp.float32)
C = cp.random.randn(N, N, dtype=cp.float32)

alpha = 2.0
beta = 0.5

# Define fused operation
@nvtx.annotate("Fused operation", color="green")
@cp.fuse()
def fused_op(A, B, C, alpha, beta):
    tmp =  alpha * A * B + beta * C
    # cp.cuda.Device(0).synchronize()
    return tmp

# Non-fused version
@nvtx.annotate("Non-fused operation", color="red")
def non_fused_op(A, B, C, alpha, beta):
    tmp = alpha * A * B + beta * C  # one kernel
    # cp.cuda.Device(0).synchronize()
    return tmp

def main():
    # Warm-up GPU (important for fair timing)
    fused_op(A, B, C, alpha, beta)
    non_fused_op(A, B, C, alpha, beta)
    cp.cuda.Device(0).synchronize()

    # Timing fused
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    start_event.record()
    _ = fused_op(A, B, C, alpha, beta)
    end_event.record()
    end_event.synchronize()
    fused_time = cp.cuda.get_elapsed_time(start_event, end_event)  # ms

    # Timing non-fused
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    start_event.record()
    _ = non_fused_op(A, B, C, alpha, beta)
    end_event.record()
    end_event.synchronize()
    non_fused_time = cp.cuda.get_elapsed_time(start_event, end_event)  # ms

    # Result
    print(f"Fused time: {fused_time:.3f} ms")
    print(f"Non-fused time: {non_fused_time:.3f} ms")
    print(f"Speedup: {non_fused_time / fused_time:.2f}x faster")

if __name__ == "__main__":
    main()
