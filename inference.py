import os
import time
import psutil
from typing import Generator, Dict, Any
from llama_cpp import Llama

class ADTCInferenceEngine:
    def __init__(self, model_path: str, context_window: int = 4096):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        self.model_path = model_path
        self.context_window = context_window
        self.threads = max(1, min(4, os.cpu_count() // 2 if os.cpu_count() else 4))
        
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.context_window,
            n_threads=self.threads,
            n_batch=512,
            f16_kv=True,
            verbose=False
        )

    def get_ram_usage(self) -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)

    def generate_reasoning_response(self, prompt: str, system_prompt: str = "") -> Generator[Dict[str, Any], None, None]:
        full_prompt = f"<|im_start||system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        start_time = time.perf_counter()
        token_count = 0
        
        response_stream = self.llm(
            prompt=full_prompt,
            max_tokens=2048,
            temperature=0.6,
            top_p=0.95,
            stream=True
        )
        
        for chunk in response_stream:
            text = chunk['choices'][0]['text']
            if text:
                token_count += 1
                elapsed = time.perf_counter() - start_time
                tps = token_count / elapsed if elapsed > 0 else 0.0
                yield {
                    "token": text,
                    "tps": tps,
                    "elapsed_time": elapsed,
                    "ram_mb": self.get_ram_usage()
                }
