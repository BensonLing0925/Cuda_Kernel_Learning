"""
ncu compile command: ncu --set basic -k matmul_reg --launch-count 1 python ncu_reg_float4_profile.py
output to text, use shell pipeline :  > tiled_profile_result.txt
"""

import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
import torch
from torch.utils.cpp_extension import load_inline
# register tiled matmul + float4 data transfer
# vectorize output
register_tiled_cuda_src = r'''
#define BM 64        // 一個 block 算 C 的 64 列
#define BN 64        // 一個 block 算 C 的 64 行
#define BK 8         // 沿 K 一次處理 8
#define TM 8         // 每個 thread 算直向 8 格

__global__ void matmul_reg(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    __shared__ float As[BM * BK];   // 64×8
    __shared__ float Bs[BK * BN];   // 8×64

    const int cRow = blockIdx.y;    // 這個 block 負責 C 的哪一條 64 列
    const int cCol = blockIdx.x;    // 哪一條 64 行

    const int threadCol = threadIdx.x % BN;        // 0..63：這 thread 的 column
    const int threadRow = threadIdx.x / BN;        // 0..7 ：負責第幾組 TM

    // 512 個 thread 協作把 A、B 的塊載進 shared，每 thread 各載一格
    const int innerColA = threadIdx.x % BK;        // 0..7
    const int innerRowA = threadIdx.x / BK;        // 0..63
    const int innerColB = threadIdx.x % BN;        // 0..63
    const int innerRowB = threadIdx.x / BN;        // 0..7

    float threadResults[TM] = {0.0f};   // ← TM 個累加器，全在 register，跨 bk 不歸零

    // 外層：沿 K 一塊塊掃（shared tiling，Day2 那套）
    for (int bk = 0; bk < K; bk += BK) {
        // 載入 A 的 64×8 塊、B 的 8×64 塊進 shared

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

        __syncthreads();

        // ===== Register tiling =====
        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
        	float tmpB = Bs[dotIdx * BN + threadCol];     // ← B 值載進 register 一次
        	for (int resIdx = 0; resIdx < TM; resIdx++) {
        		threadResults[resIdx] +=
        		As[(threadRow*TM + resIdx)*BK + dotIdx] * tmpB;   // ← tmpB 重複用 TM 次
        	}
        }
        // ===========================

        __syncthreads();
    }

    // 寫回：每個 thread 寫它負責的 TM 格
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
