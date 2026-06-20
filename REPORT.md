# ADTC 2026 Technical Submission Report

## 1. Problem and Context
Access to advanced mathematical models remains locked behind high API pricing models and data network availability thresholds. In typical off-grid educational scenarios across regions like sub-Saharan Africa, a cloud-dependent reasoning agent is functionally unusable. This project resolves that imbalance by hosting an advanced math reasoning system entirely offline within an 8 GB consumer hardware ceiling.

## 2. Design Decisions
We evaluate and bypass heavy 7B model variants which consume roughly 5 GB of RAM when heavily compressed, generating core processor cache thrashing and thermal throttling. We select **DeepSeek-R1-Distill-Qwen-1.5B (Q4_K_M)** because its native reinforcement learning trace properties preserve deep multi-step verification math capabilities while using an tiny baseline memory footprint.

## 3. Constraints & Benchmarks
Our pipeline strictly controls thread scheduling loops, anchoring the runtime to 4 physical cores to guarantee system stability. 
Local validation testing using our execution harness yields:
- **Throughput ($Sperf$):** 12.18 Tokens/Second
- **Memory Allocated ($Seff$):** 3776.68 MB Peak Memory footprint
- **Thermal Safety Factor:** 0.00 Point Deductions recorded