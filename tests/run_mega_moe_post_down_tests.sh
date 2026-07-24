#!/usr/bin/env bash
set -euo pipefail

common=(
  tests/test_mega_moe.py
  --hidden 1024
  --intermediate-hidden 1024
  --num-experts 8
  --activation swiglu
  --activation-clamp 10
  --num-correctness-tests 1
  --correctness-only
)

# Canonical, versioned legacy PRE_DOWN bit-pattern regression.
python "${common[@]}" \
  --mma-type fp8xfp4 \
  --num-processes 1 \
  --num-max-tokens-per-rank 384 \
  --num-tokens 64 \
  --num-topk 1 \
  --routing balanced \
  --route-weight-mode pre_down \
  --save-l1-preact \
  --no-save-forward-stages \
  --check-predown-golden

# Multi-rank expanded-backward matrix:
# - both BF16 and native MXFP4
# - PRE_DOWN and POST_DOWN
# - every combine order
# - BLOCK_M=16 (16 tokens) and BLOCK_M=32 (24 tokens)
# - masked/skew routes and empty-rank/duplicate-destination routes
for mma_type in bf16xbf16 fp8xfp4; do
  for route_mode in pre_down post_down; do
    for combine_mode in fixed_topk deepep deepep_v1; do
      # expected_tokens_per_expert = 16 * 2 ranks * topk 2 / 8 = 8
      # selects BLOCK_M=16. Skew + masks cover invalid fixed slots.
      python "${common[@]}" \
        --mma-type "${mma_type}" \
        --num-processes 2 \
        --num-max-tokens-per-rank 384 \
        --num-tokens 16 \
        --num-topk 2 \
        --expect-block-m 16 \
        --routing skew \
        --masked-ratio 0.25 \
        --route-weight-mode "${route_mode}" \
        --combine-order-mode "${combine_mode}" \
        --active-saved-pool \
        --save-l1-preact \
        --test-backward

      # expected_tokens_per_expert = 24 * 2 ranks * topk 2 / 8 = 12
      # selects BLOCK_M=32. Extreme routing gives repeated destination ranks;
      # rank 1 is empty but still receives and publishes remote rows.
      python "${common[@]}" \
        --mma-type "${mma_type}" \
        --num-processes 2 \
        --num-max-tokens-per-rank 384 \
        --num-tokens 24 \
        --zero-rank 1 \
        --num-topk 2 \
        --expect-block-m 32 \
        --routing extreme \
        --masked-ratio 0.25 \
        --route-weight-mode "${route_mode}" \
        --combine-order-mode "${combine_mode}" \
        --active-saved-pool \
        --save-l1-preact \
        --test-backward
    done
  done
done
