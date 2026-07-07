"""
ncu compile command: ncu --set basic -k matmul_reg --launch-count 1 python ncu_reg_float4_profile.py
output to text, use shell pipeline :  > ncu_reg_float4_profile_result.txt
"""

import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
import torch
from torch.utils.cpp_extension import load_inline
# register tiled matmul + float4 data transfer (vectorize loading)
register_tiled_cuda_src = r'''
#define BM 64 
#define BN 64
#define BK 8
#define TM 8

__global__ void matmul_reg(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    __shared__ float As[BM * BK];   // 64×8
    __shared__ float Bs[BK * BN];   // 8×64

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadCol = threadIdx.x % BN;        // 0..63
    const int threadRow = threadIdx.x / BN;        // 0..7

    const int innerColA = threadIdx.x % BK;        // 0..7
    const int innerRowA = threadIdx.x / BK;        // 0..63
    const int innerColB = threadIdx.x % BN;        // 0..63
    const int innerRowB = threadIdx.x / BN;        // 0..7

    float threadResults[TM] = {0.0f};

    for (int bk = 0; bk < K; bk += BK) {

		// using threadIdx.x < 128 to avoid warp divergence
		if (threadIdx.x < 128) {
			const int loadColA = threadIdx.x % 2;
			const int loadRowA = threadIdx.x / 2;
			float4 valA = reinterpret_cast<const float4*>(A)[((cRow*BM + loadRowA)*K + (bk + loadColA * 4)) / 4];
			As[loadRowA * BK + loadColA * 4 + 0] = valA.x;
			As[loadRowA * BK + loadColA * 4 + 1] = valA.y;
			As[loadRowA * BK + loadColA * 4 + 2] = valA.z;
			As[loadRowA * BK + loadColA * 4 + 3] = valA.w;

			const int loadColB = threadIdx.x % 16;
			const int loadRowB = threadIdx.x / 16;
			float4 valB = reinterpret_cast<const float4*>(B)[((bk + loadRowB)*N + (cCol*BN + loadColB * 4)) / 4];
        	Bs[loadRowB * BN + loadColB * 4 + 0] = valB.x;
        	Bs[loadRowB * BN + loadColB * 4 + 1] = valB.y;
        	Bs[loadRowB * BN + loadColB * 4 + 2] = valB.z;
        	Bs[loadRowB * BN + loadColB * 4 + 3] = valB.w;
		}

	/*
		// while these two if-segments theoretically decrease the amount of transcation but
		// this caused warp divergence, increasing the overall cycles used 
		if (innerColA % 4 == 0) {
			float4 valA = reinterpret_cast<const float4*>(A)[((cRow*BM + innerRowA)*K + (bk + innerColA)) / 4];
			As[innerRowA * BK + innerColA + 0] = valA.x;
			As[innerRowA * BK + innerColA + 1] = valA.y;
			As[innerRowA * BK + innerColA + 2] = valA.z;
			As[innerRowA * BK + innerColA + 3] = valA.w;
		}

		if (innerColB % 4 == 0) {
			float4 valB = reinterpret_cast<const float4*>(B)[((bk + innerRowB)*N + (cCol*BN + innerColB)) / 4];
        	Bs[innerRowB * BN + innerColB + 0] = valB.x;
        	Bs[innerRowB * BN + innerColB + 1] = valB.y;
        	Bs[innerRowB * BN + innerColB + 2] = valB.z;
        	Bs[innerRowB * BN + innerColB + 3] = valB.w;
		}
	*/

		

        // As[innerRowA * BK + innerColA] = A[(cRow*BM + innerRowA)*K + (bk + innerColA)];
        // Bs[innerRowB * BN + innerColB] = B[(bk + innerRowB)*N + (cCol*BN + innerColB)];
        __syncthreads();

        // ===== Register tiling =====
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
        	float tmpB = Bs[dotIdx * BN + threadCol];
        	for (int resIdx = 0; resIdx < TM; resIdx++) {
        		threadResults[resIdx] +=
        		As[(threadRow*TM + resIdx)*BK + dotIdx] * tmpB;
        	}
        }
        // ===========================

        __syncthreads();
    }

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
