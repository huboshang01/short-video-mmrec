#!/usr/bin/env bash
# V5-LoRA Step 04: 在 AutoDL 或本地 GPU 环境启动 LLaMA-Factory LoRA SFT。
#
# 单卡：
#   bash scripts/v5_lora/04_train_lora.sh
#
# 单机多卡：
#   CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 NPROC_PER_NODE=2 bash scripts/v5_lora/04_train_lora.sh

set -euo pipefail

CONFIG_PATH="${1:-configs/v5_lora/qwen2_5_vl_lora_sft.yaml}"

llamafactory-cli train "${CONFIG_PATH}"
