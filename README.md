# Cuda_Kernel_Learning
A would-be graduate tries to learn how to write fast kernels

- Day 1: naive CUDA matmul, verified correct vs torch (max err 1.5e-5) on T4
- Day 2: tiled CUDA matmul, verified and compared between naive and tiled (342.1 to 782.1 GFLOP/s on T4)
- Day 3: use NCU to profile tiled matrix multiplications. Bottleneck is L1/shared at 96%, DRAM only 22% (tiling works)
- Day 4: implemented 1D register tiling, 2.8x speedup over tiled version, occupancy dropped to 61% due to register pressure
- Day 5/6: attempted to vectorize load operations using float4. Discovered overall execution latency increase (from 1.36ms to 1.45ms). For more detailed debug process, see NOTES.md in Day6
