#!/usr/bin/env bash
# Verified with one LoRA inference process alongside duck_full_v1 training.
source /root/autodl-tmp/piper-training/env.sh
unset XLA_PYTHON_CLIENT_MEM_FRACTION
unset TF_GPU_ALLOCATOR
export XLA_CLIENT_MEM_FRACTION=0.20
export XLA_PYTHON_CLIENT_ALLOCATOR=bfc
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_UNIFIED_MEMORY=false
export CUDA_VISIBLE_DEVICES=0
