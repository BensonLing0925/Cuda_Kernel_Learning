import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
import torch
from torch.utils.cpp_extension import load_inline
# tiled matmul
tiled_cuda_src = r'''
#define TILE 16

__global__ void matmul_tiled(const float* A, const float* B, float* C,
              int M, int N, int K) {
    __shared__ float As[TILE][TILE];  //using shared memory and not HBM
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Each thread loads one element into shared memory.
        As[threadIdx.y][threadIdx.x] = A[row * K + (t * TILE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[col + (t * TILE + threadIdx.y) * N];

        __syncthreads();   // wait for loading all data in the block

        for (int k = 0; k < TILE; k++)
          acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();   // wait for all threads in the block
    }

    if (row < M && col < N)
      C[row * N + col] = acc;
}

torch::Tensor tiled_matmul_cuda(torch::Tensor a, torch::Tensor b) {
    int M = a.size(0), K = a.size(1), N = b.size(1);
    auto c = torch::empty({M, N}, a.options());
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matmul_tiled<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
    return c;
}
'''

tiled = load_inline(name="matmul_tiled", cpp_sources="torch::Tensor tiled_matmul_cuda(torch::Tensor, torch::Tensor);",
                  cuda_sources=tiled_cuda_src,extra_cuda_cflags=["-arch=sm_86"],functions=["tiled_matmul_cuda"], verbose=True)
M, K, N = 1024, 1024, 1024
A = torch.randn(M, K, device="cuda")
B = torch.randn(K, N, device="cuda")
cuda_out = tiled.tiled_matmul_cuda(A, B)
torch.cuda.synchronize()
