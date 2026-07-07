# Day5/6: float4 vectorization load experiments

## Goal
Change the A/B load into float4 read operations

## Process
1. Using (innerCol % 4 == 0) to filter threads to load
   Result: Max error close to 0 but execution became slower (from 1.36ms to 1.69ms)
2. Assumed warp divergence caused the slow down
   Validate: smsp__thread_inst_executed_per_inst_executed.ratio
   Comparison: Day4 has no warp divergence (32), Day5 has divergence (27.04)
3. Used (threadIdx.x < 128) to let the first 128 threads do the loading
   Result: Warp divergence solved, but overall execution time is still slower than Day4 (1.45ms)
4. New hypothesis: memory latency not being hidden
   Validate: long_scoreboard metric
   Result: Day5 has lower global memory latency stall than Day4 (6.27% vs 18.93%) -> hypothesis rejected
5. Barrier stall
   Comparison: Day4: 9.42%, Day5: 30.91%
   Reason: 128 threads have to do 4x the work compared to the original 512-thread design.
   The remaining 384 threads finish early and wait idle at __syncthreads(),
   while the 128 loading threads take longer in absolute time despite float4's per-byte efficiency gain.

## Conclusion
While float4 does increase the efficiency of data movement (proven by the lower long_scoreboard stall),
concentrating work onto fewer threads increases the absolute time each thread takes to finish.
Since __syncthreads() waits for all threads in the block to reach the barrier — determined by the
slowest thread, not the average or sum — the added barrier stall cost exceeds the benefit gained
from vectorized loading.
