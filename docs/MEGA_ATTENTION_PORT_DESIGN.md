# MegaAttention sparse-prefill port onto `nv_dev`

## Status

Implemented and validated in this checkout. The target base is
`origin/nv_dev` at `b7643bd54521f563b839b98289b5cd048c062ba2`. All changed
numerical CUDA code is isolated below `csrc/mega_attention/`; no upstream
kernel or parameter file is modified.

The source feature stack is the five-commit range after
`a8f794d1251cbfd88a5011445dd5582289c727e4` in
`subwave_hca_copy/mega_attention_flashmla`:

| Commit | Content | Required by `mega_attention_bench` |
|---|---|---|
| `9b1b56b` | caller-owned output, raw-Q, fused-O, bounded/persistent prefill, ready flags, dual KV pool, decode-side ready/raw-Q/fused-O, stable-ABI/build changes | Prefill caller-owned output, raw-Q, fused-O and `num_sms` are required. Ready flags and dual-pool are useful members of the same prefill source closure. Decode changes are not used by the current bench but are included in the
private extension for future use. Stable-ABI changes are intentionally not
ported because `nv_dev` uses pybind. |
| `b8e2af2` | standalone `mega_fwd` SM100 head-128 D=512 topology and `mega=True` dispatch | Required for the optional `mega_attn` benchmark configuration. |
| `30daf92` | split feature documentation | Documentation only. |
| `04ab686` | documentation rewrite | Documentation only. |
| `eabafc6` | consolidate documentation into `MEGA_ATTENTION.md` | Documentation only. |

## Why the commits must not be replayed

The source fork is based on the ABI-stable FlashMLA line.  The target `nv_dev`
branch uses the ordinary pybind extension and additionally supports
`indexer_topk=1024` and head-128 `lse_indexer`.  Replaying `9b1b56b` would:

1. replace or conflict with the target pybind API;
2. modify the shared `SparseAttnFwdParams` ABI;
3. fork the upstream small-topK `phase1.cuh` in place;
4. mix bench-only prefill work with unrelated sparse-decode changes; and
5. make every future upstream update to the stock kernel a three-way merge.

The CUTLASS submodule pin is identical on source and target
(`147f5673d0c1c3dcf66f78d677fd647e4a020219`).  The device-side kerutils files
used by the kernel and the stock sparse `common_subroutine.h` are also
compatible; the kerutils tree difference is confined to ABI-specific
supplemental Torch helpers.  Therefore the numerical CUDA source can be
isolated without importing the source fork's Torch ABI.

## Goals

- Preserve the stock `nv_dev` CUDA kernels and `indexer_topk` behavior.
- Provide the exact prefill call surface consumed by `mega_attention_bench`:
  caller-owned `out`, `num_sms`, raw-Q, fused-O, and `mega=True`.
- Keep the prefill-ready and dual-pool capabilities that are already part of
  the accepted source closure, without adding them to upstream parameter
  structs.
- Put all changed numerical kernel code in newly added files.
- Make the advanced extension lazy and opt-in so a normal FlashMLA import and
  upstream tests do not depend on it being loaded.
- Reproduce source-fork correctness and the measured benchmark gain on B200 or
  GB200.

## Non-goals for this port

- Do not port the ABI-stable Torch registration layer; `nv_dev` remains pybind.
- Do not change the stock sparse-decode implementation or scheduler ABI.
- Include the source fork's decode template changes in the private kernel
  copy. Expose the self-contained row-ready mode (including `num_sm_parts`,
  raw-Q and fused-O) as a separate lazy API; keep split-KV instantiated for
  a future private metadata/combine adapter.
- Do not combine `indexer_topk > 0` with the advanced kernel in this revision.
  Existing `indexer_topk` calls continue through the stock `nv_dev` extension.

## File layout

All private C++/CUDA types and kernels live under one new source root:

```text
csrc/mega_attention/
  api.cpp                         # pybind module for the private extension
  params.h                        # private prefill/decode parameter ABI
  sparse_attention.h              # prefill + ready-decode validation/dispatch
  fwd_for_small_topk/
    head128/
      config.h
      phase1.h
      phase1.cuh
      instantiations/
        phase1_prefill_k512.cu    # fork stock topology + raw-Q/fused-O
        phase1_decode_k512.cu     # split-KV + row-ready decode templates
  mega_fwd/
    PROVENANCE.md
    common_subroutine.h
    config.h
    defines.h
    helpers.h
    sparse_attention.cu
    sparse_attention.cuh
    sparse_attention.h
    utils.h

flash_mla/mega_attention_interface.py
tests/test_mega_attention_private_extension.py
```

`fwd_for_small_topk/head128` preserves the source fork's original directory
name below the private source root. It contains a private copy of the accepted
modified small-topK head-128 prefill kernel under namespace
`mega_attention::fwd_for_small_topk::head128`. This is necessary because raw-Q
and fused-O are interleaved with Q staging and the TMEM epilogue; extracting
them into a thin wrapper around the upstream kernel is not possible.

`mega_fwd` is copied as a directory, then adapted only at its private params and
launch boundary.  Its numerical body remains unchanged.  Both private kernel
families reuse the target CUTLASS pin and the unchanged device kerutils.

## Extension and Python API

Build a second extension named `flash_mla.mega_attention_cuda`.  It exposes a prefill op with the source-fork argument set and a standalone
row-ready decode op. Prefill returns `(out, max_logits, lse)`; ready-decode
returns `(out, lse)`.

The existing public Python function keeps the `nv_dev` positional API:

```python
flash_mla_sparse_fwd(
    q, kv, indices, sm_scale, d_v=512,
    attn_sink=None, topk_length=None, indexer_topk=0,
    *,
    out=None,
    num_sms=None,
    q_ready=None,
    q_ready_chunk_size=0,
    q_positions=None,
    q_cos_sin_cache=None,
    q_norm_eps=1e-6,
    o_ready=None,
    fused_o_fp8=None,
    fused_o_scale=None,
    fused_o_positions=None,
    fused_o_skip_bf16=False,
    extra_kv=None,
    extra_indices=None,
    extra_topk_length=None,
    mega=False,
)
```

Routing rules:

- No keyword-only advanced argument: call stock `flash_mla.cuda`, preserving
  all `indexer_topk` behavior.
- Any advanced argument or caller-owned `out`: lazily import and call
  `flash_mla.mega_attention_cuda`.
- Advanced path requires `indexer_topk == 0`, SM100, `h_q=128`, and
  `d_qk=d_v=512`.
- `mega=True` requires a positive even `num_sms` and rejects fused-O because
  the alternative topology does not implement that epilogue.
- Raw-Q requires `q_positions` plus `q_cos_sin_cache`.
- Fused-O requires all three output tensors plus `q_cos_sin_cache`.

The module build description should live in a new Python helper where
practical; `setup.py` should only append the returned `CUDAExtension`.  This
keeps the recurring upstream merge surface small.

## Implementation sequence

1. Add the private extension skeleton and private parameter ABI.
2. Copy `mega_fwd` unchanged, compile it in the new extension, and adapt its
   private includes/entry point.
3. Copy the modified stock small-topK kernel into its original
   `fwd_for_small_topk/head128` hierarchy below the private source root, move it
   to a private namespace, and instantiate prefill plus split/ready decode
   D=512.
4. Implement pybind validation/dispatch and caller-owned outputs.
5. Add the keyword-only Python router while leaving the normal nv_dev fast path
   byte-for-byte behaviorally unchanged.
6. Add focused correctness tests and then connect `mega_attention_bench` via
   `MEGA_ATTENTION_FLASHMLA`.

## Test matrix

### Build and upstream regression

- Build on CUDA 12.9+ for `sm_100f`/B200 or GB200.
- Import both `flash_mla.cuda` and `flash_mla.mega_attention_cuda` in one
  process to detect symbol/registration conflicts.
- Run upstream sparse-prefill tests, including head-128
  `indexer_topk=512/1024`, to prove the stock route is unchanged.

### Dedicated advanced-kernel correctness

- Private stock topology with no fusion versus stock nv_dev output.
- Caller-owned `out` identity and CUDA-graph capture/replay.
- Raw-Q versus externally RMS-normalized/RoPE-transformed Q.
- Fused-O with BF16 retained versus the standalone inverse-RoPE/FP8 oracle.
- Fused-O with `fused_o_skip_bf16=True`, validating final output projection.
- `mega=True` versus private stock output at the same inputs.
- `num_sms` bounded stock topology versus unbounded stock output.
- Ready flags and dual KV pool focused regressions.
- Row-ready decode versus the stock split-KV decode result, including
  caller-owned output, raw-Q and fused-O focused cases.
- Negative validation tests for unsupported option combinations.

### Benchmark reproduction

1. Single-layer paired output comparison at 8192 query tokens and a 64K
   prefix; require finite output and final-layer relative error below `5e-2`
   (the source measurement was about `1.4e-3`).
2. Single-layer stage timing for raw-Q, fused-O, and `mega_attn` independently.
3. Whole-model 61-layer AB/BA run with locked routing and the same lane package,
   clocks, warmup, and iteration count as the recorded result.
4. Reproduce the default stack's approximately `+4.04%` throughput gain within
   normal run-to-run noise; investigate if the result differs by more than
   0.5 percentage points on equivalent GB200 clocks.

Performance comparisons are only valid when `--probe`/`--probe-deps` reports
the same native lane implementations and the GPU clock matches the source run.

## Verification results (2026-08-03)

The extension was built from a clean `nv_dev` base on an NVIDIA B200
(148 SMs), CUDA 12.9.41, and PyTorch 2.8.0a0+nv25.05. The stock
`flash_mla.cuda` and private `flash_mla.mega_attention_cuda` modules import
in the same process without symbol or registration conflicts.

Focused regression results:

- Stock `nv_dev` `indexer_topk=512/1024` still returns the fourth
  `lse_indexer` output and matches the stock three-output result.
- Private stock-fused and `mega=True` prefill both match stock `nv_dev`
  exactly for output, max logits, and LSE; caller-owned output pointers are
  preserved.
- The source ready/raw-Q standalone cases pass at
  `(S_Q, topk)=(256,128)` and `(512,1024)`.
- The dual-KV-pool regression is exact for output, max logits, and LSE.
- Ready decode matches stock split/decode with maximum absolute error
  `0.001953125` for BF16 output and `9.54e-7` for LSE, while preserving the
  caller-owned output pointer.

The real `mega_attention_bench` loader reports the target checkout and detects
the fused-O API. At `query_len=128, prefix=4096`, its full correctness check
passes; sparse-attention relative error is `0.004254`. Paired results on this
B200 are:

- private stock versus `mega_attn`: final output relative error `0`; the
  small fixture measured `+1.00%`, close to the recorded `+0.91%`.
- standalone output quantization versus fused-O: final output relative error
  `0`; the small fixture measured `+1.90%`.
- the production attention shape `query_len=8192, prefix=65536`: final output
  relative error `0`, but `mega_attn` measured `-2.13%` on B200
  (`36.574 ms` versus `37.353 ms`). Thus functional reproduction succeeds,
  while the GB200 performance gain does not reproduce on this different
  148-SM platform.

The ported `mega_fwd` numerical body is byte-identical to the source fork;
only three private `params.h` include paths differ. A same-node build of the
original stable-ABI fork was not possible in the PyTorch 2.8 container because
that fork requires `torch/csrc/stable/tensor.h`. Full 61-layer GB200
reproduction also requires the original native lane package, CUDA/PyTorch
stable-ABI environment, 152-SM GB200 clocks, and the original routing setup.
The serial bench path is not a valid raw-Q A/B oracle because its baseline does
not run the external per-head Q RMSNorm/RoPE pass; raw-Q correctness is instead
covered by the dedicated externally preprocessed-Q regression above.

## Maintenance properties

- Upstream kernel and parameter files remain untouched.
- Private types prevent C++ ABI and CUDA symbol interposition with stock
  kernels loaded in the same process.
- The original nv_dev path remains the default and remains testable without
  exercising MegaAttention code.
- Future upstream rebases should conflict only in the extension list and the
  public Python routing function; numerical CUDA code remains in added files.
