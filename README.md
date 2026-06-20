# Cuda_Kernel_Learning
A would-be graduate tries to learn how to write fast kernels

- Day 1: naive CUDA matmul, verified correct vs torch (max err 1.5e-5) on T4
- Day 2: tiled CUDA matmul, verified and compared between naive and tiled (342.1 to 782.1 GFLOP/s on T4)
