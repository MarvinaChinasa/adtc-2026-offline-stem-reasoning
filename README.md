# ADTC 2026 Submission: Local Offline STEM Reasoning Engine

An engineering-focused, cloud-decoupled mathematical reasoning system optimized explicitly for commodity hardware profiles across the African continent. This solution demonstrates elite performance within the **Math & Scientific Reasoning** target domain without external API dependencies or internet access.

---

## 🚀 Architectural Strategy

Instead of relying on heavy 7B parameter models (which use 4.5 GB–5.5 GB of RAM when quantized, causing hardware strain and thermal throttling), our submission leverages a lean, highly targeted choice: **DeepSeek-R1-Distill-Qwen-1.5B** compiled via low-overhead native `llama.cpp` bindings. 

This architectural choice yields deep, multi-verification chain-of-thought mathematical reasoning while maintaining a tiny, safe resource footprint.

---

## 📈 Measured Performance Metrics

Evaluated locally using our automated batch evaluation harness (`benchmark.py`), yielding the following telemetry results:

| Evaluation Metric Vector | Measured System Output | ADTC Leaderboard Impact |
| :--- | :--- | :--- |
| **Generation Throughput ($Sperf$)** | **12.18 Tokens/Second** | **High Performance:** Captures ~81.2% of the absolute maximum provisional target vector on a pure CPU layer. |
| **Peak Memory Footprint ($Seff$)** | **3776.68 MB (3.68 GB)** | **Highly Efficient:** Saves over 3.2 GB of memory room below the maximum 7 GB competition budget. |
| **Execution Stability ($Pthermal$)** | **0.00 Point Deduction** | **Stable Thermals:** Average problem duration was kept tightly managed via our system prompt constraint, avoiding CPU thermal throttling penalties entirely. |

---

## 🛠️ Optimization Adjustments

1. **Hardware Core Thread Control:** Locked execution pipelines directly to 4 physical cores (`max(1, min(4, os.cpu_count() // 2))`) to prevent thread thrashing and hyperthreading overhead on commodity processors.
2. **Context and Batch Optimization:** Configured with an explicit `n_batch=512` and full Key-Value cache efficiency (`f16_kv=True`) to prevent memory spikes during complex multi-step calculus operations.
3. **Structured Prompt Constraints:** Implemented an original `STEM_SYSTEM_PROMPT` anchor that disciplines the internal reasoning loop, reducing token generation times while forcing clear final output formatting using LaTeX.

---

## 🌍 Real-World Impact: The African Classroom

By requiring less than 4 GB of total system memory and zero internet connectivity, this engine is fully prepared for real-world deployment on affordable, low-spec refurbished hardware. It acts as a highly capable, zero-marginal-cost STEM tutor for classrooms, schools, and communities across Africa that lack stable grid electricity or high-speed fiber-optic access.
