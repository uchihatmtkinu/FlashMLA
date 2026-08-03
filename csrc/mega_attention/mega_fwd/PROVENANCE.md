# Sparse-attention source closure

The accepted sources are the manifest-selected SM100 small-TopK head-128
prefill prefix and tail specializations. Their historical source files ended
in `wqb_adaptive_r11.cu` and `baseline_topology_r10.cu`. This directory
intentionally exposes no iteration name; its stable C++ entry points describe
the prefix and tail lanes semantically.

## Accepted source mapping

| Standalone file | Accepted source role |
|---|---|
| `sparse_attention.cu` | D=512 native GB200 prefix, bounded B200 prefix, and bounded tail launch selection |
| `sparse_attention.cuh` | numerical phase, raw-Q transform, ready wait, TMA and cluster launch |
| `sparse_attention.h` | stable public C++ declaration |
| `config.h` | SM100 head-128 shared/tensor-memory configuration |
| `params.h` | preallocated input/output, stride, ready-flag and raw-Q ABI |
| `common_subroutine.h` | sparse index masking, P reduction and online-softmax helpers |
| `helpers.h`, `defines.h`, `utils.h` | local scalar/vector and launch helpers |
| `kerutils/` | the small header-only helper closure required by the files above |

The numerical body is retained intact. `launch_sparse_attention_prefix`
selects the native problem-sized FlashMLA schedule on a 152-SM GB200: 6,784
two-CTA clusters, or 13,568 CTAs. On a 148-SM B200 production selects the
source-qualified schedule-10 prefix at grid 108: 54 two-CTA clusters
leave 40 SMs free, enough for the ready-aware WqB producer's steady 36-SM
phase plus a 4-SM admission margin. Schedule 11/grid148 remains available for
isolated experiments but is not the production route because child-node
priority alone cannot guarantee CTA admission. The binding requires raw-Q
metadata and 128-row ready chunks for both. `launch_sparse_attention_tail` selects the
accepted bounded tail schedule and is locked to 1,408 rows with 64 CTAs on
B200 or 68 CTAs on GB200. `launch_sparse_attention` remains the bounded
compatibility entry point. The production entry points preserve preallocated
outputs and acquire polling of `q_ready`.

## External headers

The closure still intentionally requires:

- CUDA Toolkit headers and runtime, including BF16, FP8 and tensor-map APIs;
- official CUTLASS/CuTe headers from a public upstream checkout;
- SM100a compilation with the original kernel feature define
  `KERUTILS_ENABLE_SM100A`.

It does not include or reference a FlashMLA checkout. The copied helper closure
is Apache-2.0-derived source and must remain covered by this repository's
NOTICE attribution.

## Remaining gate

Source closure and reference tests do not prove successful SM100 compilation.
The final build gate must compile this translation unit against the selected
public CUTLASS revision, verify no unresolved include/symbol remains, then run
raw-Q and pretransformed-Q correctness on GB200.
