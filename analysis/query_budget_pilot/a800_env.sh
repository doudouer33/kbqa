#!/bin/sh
# Source this file on A800. All downloaded files and runtime caches stay below /nas/luoyichen.
export QB_A800_ROOT=/nas/luoyichen
export XDG_CACHE_HOME="$QB_A800_ROOT/cache"
export HF_HOME="$QB_A800_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export SENTENCE_TRANSFORMERS_HOME="$QB_A800_ROOT/models/sentence-transformers"
export TORCH_HOME="$QB_A800_ROOT/cache/torch"
export PIP_CACHE_DIR="$QB_A800_ROOT/cache/pip"
export PYTHONPYCACHEPREFIX="$QB_A800_ROOT/cache/pycache"
export CUDA_CACHE_PATH="$QB_A800_ROOT/cache/cuda"
export TRITON_CACHE_DIR="$QB_A800_ROOT/cache/triton"
export TMPDIR="$QB_A800_ROOT/tmp"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HF_ASSETS_CACHE" "$SENTENCE_TRANSFORMERS_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TMPDIR"
