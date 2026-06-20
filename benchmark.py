import os
import json
import time
import sys
import psutil

# Ensure current working directory is in system path for local imports
sys.path.append('/kaggle/working')
from inference import ADTCInferenceEngine

class ADTCBenchmarkPipeline:
    def __init__(self, model_path: str):
        print("[*] Initializing ADTC Local Evaluation Harness...")
        self.engine = ADTCInferenceEngine(model_path=model_path)
        self.results_log = []

    def load_validation_set(self, data_path: str):
        if not os.path.exists(data_path):
            print(f"[!] Validation path {data_path} not found. Generating standardized mock math subset.")
            return [
                {"id": "math_01", "prompt": "Solve for x: 3x + 7 = 22"},
                {"id": "math_02", "prompt": "Calculate the derivative of f(x) = x^3 - 5x + 2 with respect to x."},
                {"id": "math_03", "prompt": "Find the sum of the first 20 terms of an arithmetic progression where a=3 and d=4."}
            ]
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def run_evaluation(self, validation_data, system_prompt: str):
        total_prompts = len(validation_data)
        global_start_time = time.perf_counter()
        
        peak_overall_ram = 0.0
        total_tokens_produced = 0
        total_generation_time = 0.0

        print(f"\n[+] Processing {total_prompts} Benchmark Evaluation Samples...")
        print("-" * 60)

        for idx, item in enumerate(validation_data, 1):
            prompt_id = item.get("id", f"sample_{idx}")
            prompt_text = item.get("prompt", "")
            
            print(f"[{idx}/{total_prompts}] Testing ID: {prompt_id} | Memory footprint: {self.engine.get_ram_usage():.2f} MB")
            
            output_tokens = []
            last_metrics = {"tps": 0.0, "elapsed_time": 0.0, "ram_mb": 0.0}
            
            for update in self.engine.generate_reasoning_response(prompt_text, system_prompt=system_prompt):
                output_tokens.append(update["token"])
                last_metrics = update
                if update["ram_mb"] > peak_overall_ram:
                    peak_overall_ram = update["ram_mb"]

            full_response = "".join(output_tokens)
            generated_tokens_count = len(output_tokens)
            total_tokens_produced += generated_tokens_count
            total_generation_time += last_metrics["elapsed_time"]

            self.results_log.append({
                "id": prompt_id,
                "prompt": prompt_text,
                "response": full_response,
                "metrics": {
                    "latency_seconds": round(last_metrics["elapsed_time"], 2),
                    "tokens_generated": generated_tokens_count,
                    "tps": round(last_metrics["tps"], 2),
                    "peak_ram_mb": round(last_metrics["ram_mb"], 2)
                }
            })

        global_duration = time.perf_counter() - global_start_time
        avg_tps = total_tokens_produced / total_generation_time if total_generation_time > 0 else 0.0

        summary = {
            "evaluation_metrics": {
                "total_evaluation_time_seconds": round(global_duration, 2),
                "average_tokens_per_second": round(avg_tps, 2),
                "peak_ram_consumed_mb": round(peak_overall_ram, 2),
                "efficiency_vs_7gb_budget_percent": round(((7000 - peak_overall_ram) / 7000) * 100, 2)
            },
            "detailed_runs": self.results_log
        }
        return summary

if __name__ == "__main__":
    MODEL_PATH = "/kaggle/working/models/deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf"
    VALIDATION_SET_PATH = "/kaggle/working/adtc_math_validation.json"
    OUTPUT_REPORT_PATH = "/kaggle/working/adtc_performance_report.json"
    
    STEM_SYSTEM_PROMPT = (
        "You are an optimized, offline African classroom STEM tutor. "
        "First, use your <think> tags to reason through the problem concisely. "
        "Do not repeat methods. Once verified, output your clear, final answer immediately."
    )

    pipeline = ADTCBenchmarkPipeline(model_path=MODEL_PATH)
    dataset = pipeline.load_validation_set(data_path=VALIDATION_SET_PATH)
    report_data = pipeline.run_evaluation(dataset, system_prompt=STEM_SYSTEM_PROMPT)
    
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as report_file:
        json.dump(report_data, report_file, indent=4)
        
    print("\n" + "="*60)
    print("[+] AUTOMATED BENCHMARK COMPLETE")
    print(f"[-] Final System Performance Metrics: {json.dumps(report_data['evaluation_metrics'], indent=2)}")
    print(f"[-] Comprehensive metrics file exported securely to: {OUTPUT_REPORT_PATH}")
    print("="*60)
