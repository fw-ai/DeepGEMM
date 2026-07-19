#!/usr/bin/env bash
set -euo pipefail

common=(
  tests/test_mega_moe.py
  --mma-type fp8xfp4
  --hidden 1024
  --intermediate-hidden 1024
  --num-experts 8
  --activation swiglu
  --activation-clamp 10
  --routing balanced
  --num-correctness-tests 1
  --correctness-only
)

# Canonical, versioned legacy PRE_DOWN bit-pattern regression.
python "${common[@]}" \
  --num-processes 1 \
  --num-max-tokens-per-rank 384 \
  --num-tokens 64 \
  --num-topk 1 \
  --route-weight-mode pre_down \
  --save-l1-preact \
  --no-save-forward-stages \
  --check-predown-golden

# Single-rank top-k > 1, active saved-pool padding, backward route gradients,
# and forced ring reuse (768 routed tokens in a 384-row ring).
python "${common[@]}" \
  --num-processes 1 \
  --num-max-tokens-per-rank 384 \
  --num-ring-tokens 384 \
  --require-ring-wrap \
  --num-tokens 384 \
  --num-topk 2 \
  --route-weight-mode post_down \
  --active-saved-pool \
  --save-l1-preact \
  --test-backward

# Cross-rank dispatch/combine with heterogeneous route slots and top-k > 1.
python "${common[@]}" \
  --num-processes 2 \
  --num-max-tokens-per-rank 384 \
  --num-ring-tokens 768 \
  --num-tokens 128 \
  --num-topk 2 \
  --route-weight-mode post_down \
  --active-saved-pool \
  --save-l1-preact
