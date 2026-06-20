#!/bin/bash
mkdir -p model

MODEL_URL="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
OUTPUT_PATH="model/deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf"

if [ ! -f "$OUTPUT_PATH" ]; then
    echo "[*] Downloading model file to $OUTPUT_PATH..."
    wget -O "$OUTPUT_PATH" "$MODEL_URL"
    echo "[+] Download complete!"
else
    echo "[+] Model file already exists. Skipping download."
fi