# Optional SM100 sparse-attention extension

## Scope

This change adds an optional SM100 sparse-attention extension while preserving
the existing `nv_dev` implementation and its default Python call path. The new
CUDA code is isolated below `csrc/persistent_sparse_attention/`; existing numerical kernels
and parameter structures are not modified.

The implementation combines persistent execution with kernel fusion to reduce
launch overhead and avoid unnecessary intermediate tensor traffic. Scheduling
details and application-specific pipeline configuration are intentionally kept
outside this repository-facing design note.

Build and regression instructions are in
[`PERSISTENT_SPARSE_ATTENTION_VALIDATION.md`](PERSISTENT_SPARSE_ATTENTION_VALIDATION.md).

## Design goals

- Keep the stock extension and `indexer_topk` behavior unchanged.
- Put specialized CUDA code and parameter types in newly added files.
- Use persistent execution and fused preprocessing/epilogue paths where the
  specialized interface provides the required inputs and output storage.
- Load the optional extension lazily, only when an advanced argument is used.
- Support caller-owned output, bounded launches, optional Q preprocessing,
  optional output epilogue tensors, ready flags, and an additional KV pool.
- Include the corresponding row-ready decode specialization for future use.
- Keep the recurring upstream merge surface limited to build registration and
  Python routing.

## Source layout

```text
csrc/persistent_sparse_attention/
  api.cpp
  params.h
  sparse_attention.h
  fwd_for_small_topk/head128/
    config.h
    phase1.h
    phase1.cuh
    instantiations/
      phase1_prefill_k512.cu
      phase1_decode_k512.cu
  psa_fwd/
    sparse_attention.cu
    sparse_attention.cuh
    sparse_attention.h
    ...

flash_mla/persistent_sparse_attention_interface.py
persistent_sparse_attention_build.py
tests/test_persistent_sparse_attention_private_extension.py
```

`fwd_for_small_topk/head128` is a private specialization of the upstream
small-topK kernel. The optional preprocessing and epilogue operations are
interleaved with Q staging and the TMEM epilogue, so they cannot be implemented
as a thin wrapper around the stock kernel.

`psa_fwd` is a second private prefill implementation. Both kernel families
use private namespaces and parameter types, and reuse the repository's CUTLASS
revision and device utilities.

## Extension and routing

The specialized sources build as `flash_mla.persistent_sparse_attention_cuda`, separate
from `flash_mla.cuda`. The private prefill API returns
`(out, max_logits, lse)` and the row-ready decode API returns `(out, lse)`.

The public `flash_mla_sparse_fwd` positional API remains unchanged. Routing is
based on keyword-only options:

- Calls without advanced options continue through the stock extension.
- Advanced options or caller-owned output lazily load the private extension.
- Existing `indexer_topk` calls continue through the stock extension.
- The private path is restricted to its compiled SM100/head/dimension shape.
- Unsupported option combinations fail during argument validation rather than
  silently selecting another kernel.

The extension build definition lives in `persistent_sparse_attention_build.py`; `setup.py`
only registers the returned `CUDAExtension`. CUDA 13.x builds include the
toolkit's `include/cccl` directory explicitly.

## Validation strategy

The focused regression checks four boundaries:

1. the unmodified stock route still produces the expected result;
2. both private prefill implementations match the stock numerical result;
3. caller-owned storage and private dispatch behave as specified; and
4. row-ready decode matches the stock decode result within the documented
   floating-point tolerance.

Source-to-port checks additionally compare generated device code and runtime
under a matched compiler, GPU, clock, and input setup. This distinguishes port
overhead from differences caused by hardware or benchmark configuration.

See [`PERSISTENT_SPARSE_ATTENTION_VALIDATION.md`](PERSISTENT_SPARSE_ATTENTION_VALIDATION.md) for the
publicly reproducible test command and summarized results.

## Maintenance properties

- Existing CUDA kernel and parameter files remain untouched.
- Private types and namespaces prevent ABI and symbol collisions when both
  extensions are loaded in one process.
- A normal import and the default sparse-attention path do not require the
  optional extension to be loaded.
- Future upstream rebases should only touch extension registration or Python
  routing; the specialized numerical code remains in added files.
