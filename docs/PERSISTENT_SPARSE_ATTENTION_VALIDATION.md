# SM100 extension validation

This document provides a repository-local correctness check for the optional
SM100 sparse-attention extension. It intentionally focuses on the information
needed to review and maintain this change.

## Requirements

- an SM100 GPU;
- CUDA 12.9 or newer;
- a CUDA-enabled PyTorch build compatible with the selected toolkit; and
- the `csrc/cutlass` submodule initialized.

CUDA 13.x is supported by adding `${CUDA_HOME}/include/cccl` in both extension
build definitions; no additional user flag is required.

## Quick reproduction

From a clean checkout:

```bash
git submodule update --init csrc/cutlass

FLASH_MLA_DISABLE_SM90=1 MAX_JOBS=24 NVCC_THREADS=4 \
python3 setup.py build_ext --inplace

python3 tests/test_persistent_sparse_attention_private_extension.py
```

`FLASH_MLA_DISABLE_SM90=1` limits this build to the SM100 target exercised by
the focused test. A passing run ends with:

```text
stock indexer_topk 512 ok
stock indexer_topk 1024 ok
private out 0.0
private max 0.0
private lse 0.0
psa out 0.0
psa max 0.0
psa lse 0.0
PERSISTENT_SPARSE_ATTENTION_PRIVATE_EXTENSION_OK
```

The test also reports decode differences. Those values must satisfy the test's
`atol=2e-2, rtol=2e-2` acceptance bound.

## Coverage

`tests/test_persistent_sparse_attention_private_extension.py` verifies:

- the stock route still supports `indexer_topk=512/1024` and returns its
  `lse_indexer` output;
- the private small-topK prefill implementation matches stock output, maximum
  logits, and LSE;
- the alternative private prefill implementation matches the same reference;
- caller-owned output storage is preserved; and
- row-ready decode matches the stock decode result within tolerance.

Additional development-time checks covered optional Q preprocessing, ready
flags, an additional KV pool, and the output epilogue path.

## Results

The extension was built and tested with both CUDA 12.9 and CUDA 13.x on SM100.
In the focused prefill comparisons, output, maximum logits, and LSE were
bit-exact. Row-ready decode remained within the regression tolerance.

A matched source-to-port audit found identical generated device instruction
encodings and resource usage for the ported kernel specializations. Repeated
same-node timing comparisons were within normal measurement noise, with no
measurable migration overhead. These checks establish port equivalence; they
are not a claim of end-to-end application speedup.

The optimized path uses persistent execution and kernel fusion. The focused
test validates its numerical behavior and port equivalence without publishing
application-specific scheduling or end-to-end benchmark data.

Performance comparisons are meaningful only when compiler, GPU, clocks,
inputs, cache state, warmup, and launch configuration are held constant.

## Related files

- Design and maintenance decisions:
  `docs/PERSISTENT_SPARSE_ATTENTION_PORT_DESIGN.md`
- Focused regression:
  `tests/test_persistent_sparse_attention_private_extension.py`
- Private extension build definition:
  `persistent_sparse_attention_build.py`
- Python interface and routing:
  `flash_mla/persistent_sparse_attention_interface.py` and
  `flash_mla/flash_mla_interface.py`
