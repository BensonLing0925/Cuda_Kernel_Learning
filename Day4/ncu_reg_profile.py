"""
ncu compile command: ncu --set basic -k matmul_reg --launch-count 1 python ncu_reg_profile.py
output to text, use shell pipeline :  > reg_profile_result.txt
"""

import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
import torch
from torch.utils.cpp_extension import load_inline
# register tiled matmul
register_tiled_cuda_src = r'''
#define BM 64        // a block is 64 rows
#define BN 64        // a block is 64 cols
#define BK 8         // shared-memory tile
#define TM 8 

__global__ void matmul_reg(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    __shared__ float As[BM * BK];   // 64×8
    __shared__ float Bs[BK * BN];   // 8×64

    const int cRow = blockIdx.y;    // get the block row id
    const int cCol = blockIdx.x;    // get the block col id

    const int threadCol = threadIdx.x % BN;        // 0..63, inner coordiation of shared tile
    const int threadRow = threadIdx.x / BN;        // 0..7

	// use 512(64 x 8 and 8 x 64) threads to load elements inside shared memory
    const int innerColA = threadIdx.x % BK;        // 0..7
    const int innerRowA = threadIdx.x / BK;        // 0..63
    const int innerColB = threadIdx.x % BN;        // 0..63
    const int innerRowB = threadIdx.x / BN;        // 0..7

    float threadResults[TM] = {0.0f};   // 8 registers to accumulate answers

    // move around the interconnected dimension (K)
    for (int bk = 0; bk < K; bk += BK) {
        // load A to As, size(64, 8) and load B to Bs, size(8, 64)
		// As and Bs are shared memory
        As[innerRowA * BK + innerColA] = A[(cRow*BM + innerRowA)*K + (bk + innerColA)];
        Bs[innerRowB * BN + innerColB] = B[(bk + innerRowB)*N + (cCol*BN + innerColB)];
        __syncthreads();

        // ===== Register tiling =====
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
        	float tmpB = Bs[dotIdx * BN + threadCol];     // read one element into the register
        	for (int resIdx = 0; resIdx < TM; resIdx++) {
        		threadResults[resIdx] +=
        		As[(threadRow*TM + resIdx)*BK + dotIdx] * tmpB;   // using the same element for 8 times
        	}
        }
        // ===========================

        __syncthreads();
    }

    // write back to each block
    for (int resIdx = 0; resIdx < TM; resIdx++) {
        int r = cRow*BM + threadRow*TM + resIdx;
        int c = cCol*BN + threadCol;
        if (r < M && c < N)
            C[r*N + c] = threadResults[resIdx];
    }
}

torch::Tensor reg_matmul_cuda(torch::Tensor a, torch::Tensor b) {
    int M = a.size(0), K = a.size(1), N = b.size(1);
    auto c = torch::empty({M, N}, a.options());
    dim3 threads(BM * BN / TM);                          // 512
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_reg<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
    return c;
}
'''

reg = load_inline(name="matmul_reg", cpp_sources="torch::Tensor reg_matmul_cuda(torch::Tensor, torch::Tensor);",
                  cuda_sources=register_tiled_cuda_src,extra_cuda_cflags=["-arch=sm_86"],functions=["reg_matmul_cuda"], verbose=True)
M, K, N = 1024, 1024, 1024
A = torch.randn(M, K, device="cuda")
B = torch.randn(K, N, device="cuda")
ref = A @ B
cuda_out = reg.reg_matmul_cuda(A, B)
print("max error:", (cuda_out - ref).abs().max().item())
torch.cuda.synchronize()
