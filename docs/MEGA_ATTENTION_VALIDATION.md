# MegaAttention validation and reproduction

This document records how the isolated SM100 MegaAttention port was tested,
what each comparison uses as its baseline, and which results are reproducible
from this repository alone.

## What is being validated

The private extension contains two prefill implementations:

- `fwd_for_small_topk/head128`: the source fork's stock-derived topology with
  caller-owned output, bounded launch, raw-Q, fused-O, ready flags, and dual-KV
  support. This is the default path used by `mega_attention_bench` when
  `mega=False`.
- `mega_fwd`: the alternative two-CTA-cluster topology selected by
  `mega=True`.

It also compiles split-KV and row-ready decode specializations below
`fwd_for_small_topk/head128`. The current benchmark is prefill-only; its decode
code is included for future use and is tested separately.

Three comparisons appear below and must not be conflated:

1. **Stock regression:** official `nv_dev` versus the private extension.
2. **Port equivalence:** the source fork versus this port, built on the same
   machine with the same compiler. This detects migration overhead.
3. **Benchmark gain:** a baseline model/pipeline versus a feature-enabled
   model/pipeline. This is platform- and schedule-dependent.

## Tested environments

| GPU | CUDA | PyTorch | Coverage |
|---|---|---|---|
| NVIDIA B200, 148 SMs | 12.9.41 | 2.8.0a0+nv25.05 | clean build, focused regression, raw-Q/ready, dual-KV, fused-O, and `mega_attention_bench` integration |
| NVIDIA B200, 148 SMs | 13.2 | 2.12.0a0+nv26.05 | full stock/private build, focused regression, and same-node source-to-port SASS/runtime audit |

The CUDA 13.x build requires `${CUDA_HOME}/include/cccl`. Both extension build
definitions add that path; no extra user flag is needed.

## Quick reproduction from this repository

### Requirements

- an SM100 GPU, such as B200 or GB200;
- CUDA 12.9 or newer for SM100 compilation;
- a CUDA-enabled PyTorch build compatible with the selected CUDA toolkit;
- the `csrc/cutlass` submodule initialized.

From a clean checkout:

```bash
git submodule update --init csrc/cutlass

FLASH_MLA_DISABLE_SM90=1 MAX_JOBS=24 NVCC_THREADS=4 \
python3 setup.py build_ext --inplace

python3 tests/test_mega_attention_private_extension.py
```

`FLASH_MLA_DISABLE_SM90=1` limits generated code to the SM100 target used by
this test; it does not disable SM100 or the private extension.

A passing run ends with:

```text
stock indexer_topk 512 ok
stock indexer_topk 1024 ok
private out 0.0
private max 0.0
private lse 0.0
mega out 0.0
mega max 0.0
mega lse 0.0
decode out 0.001953125
decode lse 9.5367431640625e-07
MEGA_ATTENTION_PRIVATE_EXTENSION_OK
```

The exact decode error can vary slightly with the toolchain, but must remain
within the test's `atol=2e-2, rtol=2e-2` acceptance bound.

### What the focused regression covers

`tests/test_mega_attention_private_extension.py` verifies:

- the unmodified stock route still supports `indexer_topk=512/1024` and its
  fourth `lse_indexer` output;
- private `fwd_for_small_topk/head128` prefill matches stock output, max logits,
  and LSE;
- `mega_fwd` prefill matches the same stock result;
- caller-owned output storage is preserved;
- row-ready decode matches official stock split/decode within tolerance.

On both tested B200 environments, the two private prefill paths were exact for
output, max logits, and LSE. The row-ready decode comparison against official
stock produced maximum absolute errors of `0.001953125` for BF16 output and
`9.54e-7` for LSE.

## Minimal `mega_attention_bench` reproduction

The companion `mega_attention_bench` checkout discovers FlashMLA through
`MEGA_ATTENTION_FLASHMLA`. It also requires its DeepGEMM checkout. Set both
paths explicitly so the loader cannot silently select another installation:

```bash
cd /path/to/mega_attention_bench

export MEGA_ATTENTION_FLASHMLA=/path/to/FlashMLA
export MEGA_ATTENTION_DEEP_GEMM=/path/to/mega_attention_deep_gemm
```

Both native extensions must be built for the active CUDA/PyTorch ABI. For
example, loading a CUDA 12 DeepGEMM binary in the CUDA 13.2 environment fails
with `ImportError: libcudart.so.12`; rebuild the companion checkout before
running the probe rather than changing the FlashMLA result.

First inspect dependency resolution. `--probe-deps` exits after reporting the
selected implementations:

```bash
python3 v4_attention_layer.py \
  --prefill --batch 1 --query-len 128 --prefix 4096 \
  --index-kernel --causal-index --fp4-index \
  --probe-deps
```

Then compare the default private topology against raw-Q plus fused-O:

```bash
python3 v4_attention_layer.py \
  --prefill --batch 1 --query-len 128 --prefix 4096 \
  --index-kernel --causal-index --fp4-index \
  --pair-with raw_q,fused_o --compare-output \
  --warmup 3 --iters 10
```

To exercise `mega_fwd`, use `mega_attn` without fused-O:

```bash
python3 v4_attention_layer.py \
  --prefill --batch 1 --query-len 128 --prefix 4096 \
  --index-kernel --causal-index --fp4-index \
  --pair-with mega_attn --compare-output \
  --warmup 3 --iters 10
```

The acceptance criteria are finite outputs and `final_output rel < 5e-2`.
Timing from this small fixture is a smoke test, not the GB200 headline result.

### Observed B200 integration results

At `query_len=128, prefix=4096`, the benchmark loader reported this checkout,
detected the fused-O API, and passed its correctness check. The sparse-attention
relative error was `0.004254`.

Paired results on the same 148-SM B200 were:

| Comparison | Final-output result | Measured change |
|---|---:|---:|
| private `fwd_for_small_topk/head128` versus `mega_fwd` | relative error `0` | small fixture: `+1.00%` |
| standalone output quantization versus fused-O | relative error `0` | small fixture: `+1.90%` |
| production attention shape, stock-derived versus `mega_fwd` | relative error `0` | `mega_fwd`: `-2.13%` |

The production attention shape was `query_len=8192, prefix=65536`; the
stock-derived path measured `36.574 ms` and `mega_fwd` measured `37.353 ms`.
Thus functional reproduction passed, but the source GB200 `mega_fwd` gain did
not reproduce on this different 148-SM platform.

## Same-node source-to-port equivalence audit

The original stable-ABI source fork and this pybind port were built with NVCC
13.2 on the same B200. The audit normalized private namespace/symbol names and
then compared device instruction encodings and resource usage.

### Device-code equivalence

| Kernel family | Functions | Instructions | Result |
|---|---:|---:|---|
| `fwd_for_small_topk/head128` prefill | 2 | 16,032 | all instruction encodings identical |
| split/ready decode | 2 | 16,304 | all instruction encodings identical |
| `mega_fwd` | 13 | 55,664 | all instruction encodings identical |

Registers, shared memory, barriers, stack frames, and spills also matched.
A subsequent directory/namespace-only refactor restored the source hierarchy.
The cached prefill and decode objects were compared before and after that
refactor: all 32,064 and 32,608 64-bit encoding words, respectively, remained
identical.

### Focused runtime parity

All measurements below compare the source fork directly with this port. A
negative delta means the port was slightly faster; differences around one
percent are normal launch/timing noise.

| Case | Source fork | Port | Port delta | Precision |
|---|---:|---:|---:|---|
| bounded/persistent stock-derived prefill | 22.446 us | 22.152 us | -1.31% | bit-exact |
| `mega_fwd` prefill | 24.550 us | 24.405 us | -0.59% | bit-exact |
| raw-Q plus ready flags | 161.622 us | 161.202 us | -0.26% | bit-exact |
| dual-KV pool | 23.027 us | 22.707 us | -1.39% | bit-exact |
| fused-O | 40.559 us | 40.137 us | -1.04% | BF16/max/LSE/FP8/packed scale bit-exact |

For the production direct-attention shape
`S_Q=8192, S_KV=65536, topk=1024, num_sms=116`, with the B200 SM clock locked
to 1800 MHz:

| Topology | Source fork | Port | Port delta | Precision |
|---|---:|---:|---:|---|
| `fwd_for_small_topk/head128` | 2423.228 us | 2423.092 us | -0.006% | bit-exact |
| `mega_fwd` | 2466.110 us | 2463.158 us | -0.120% | bit-exact |

This is the migration-loss test: the port adds no measurable prefill overhead.
It is not a claim that `mega_fwd` is faster than the stock-derived topology.

### Decode wrapper difference

Decode device SASS is identical, but the host wrappers are not one-for-one:

- source public row-ready call without cached metadata: `61.757 us`;
- port direct row-ready call: `47.087 us`;
- source call with cached metadata: `47.456 us`;
- port direct row-ready call in the same comparison: `47.082 us` (`-0.79%`).

The uncached source wrapper generates scheduler metadata and allocates split-KV
accumulation workspace even in row-ready mode. The private port dispatches the
self-contained row-ready specialization directly. This intentional wrapper
difference explains the apparent `23.75%` improvement.

Source-versus-port decode precision was:

- BF16 output maximum absolute error: `0.0009765625`;
- LSE maximum absolute error: `4.768e-7`;
- fused scale: bit-exact;
- FP8 output: 174 differing bytes out of 4,194,304.

Because device SASS is identical, this small difference comes from the host
wrapper/TMA parameter path rather than a changed numerical kernel.

## Full-model GB200 reproduction

The source headline used the default feature stack:

```text
raw_q,fused_o,mega_frontend
```

This path uses `fwd_for_small_topk/head128`, not `mega_fwd`. `mega_frontend`
belongs to the companion DeepGEMM repository.

A full reproduction additionally requires the original native lane-kernel
package, the matching DeepGEMM checkout, locked routing, and comparable GB200
clocks. Always run the probe first:

```bash
cd /path/to/mega_attention_bench

MEGA_ATTENTION_FLASHMLA=/path/to/FlashMLA \
MEGA_ATTENTION_DEEP_GEMM=/path/to/mega_attention_deep_gemm \
python3 model_v4_pro.py --layers 61 --prefix 131072 --lock-routing --probe
```

One production-shaped point:

```bash
MEGA_ATTENTION_FLASHMLA=/path/to/FlashMLA \
MEGA_ATTENTION_DEEP_GEMM=/path/to/mega_attention_deep_gemm \
python3 model_v4_pro.py \
  --layers 61 --query-len 8192 --prefix 131072 \
  --opts raw_q,fused_o,mega_frontend \
  --lock-routing --warmup 3 --iters 20
```

The sweep behind the source headline is:

```bash
MEGA_ATTENTION_FLASHMLA=/path/to/FlashMLA \
MEGA_ATTENTION_DEEP_GEMM=/path/to/mega_attention_deep_gemm \
python3 model_v4_pro.py \
  --layers 61 --query-len 8192 --lock-routing \
  --opts raw_q,fused_o,mega_frontend \
  --prefix-sweep 0 32768 65536 98304 131072 163840 196608 \
  --warmup 3 --iters 20 --json results/reproduction.json
```

The recorded source run used a GB200 NVL72 at 2062 MHz, 8192-token chunks,
61 layers, and locked routing:

| Metric | Baseline | Feature stack | Change |
|---|---:|---:|---:|
| 200K-token prefill wall time | 16.102 s | 15.477 s | -3.88% |
| throughput | 12,719 tok/s | 13,232 tok/s | +4.04% |

These numbers compare the vLLM-schedule baseline with the complete feature
stack. They do not measure source-to-port overhead. The alternative
`mega_fwd` topology was recorded as approximately `+0.91` model-level points
over the stock-derived topology when fused-O was disabled on both sides.

Do not compare performance across different GPU models, SM counts, clocks,
native lane implementations, cache state, warmup counts, or routing. A large
source-versus-port timing difference under matched conditions indicates a port
problem; a GB200-versus-B200 difference does not.

## Related files

- Port architecture and maintenance decisions:
  `docs/MEGA_ATTENTION_PORT_DESIGN.md`
- Kernel source mapping and provenance:
  `csrc/mega_attention/mega_fwd/PROVENANCE.md`
- Reproducible focused regression:
  `tests/test_mega_attention_private_extension.py`
- Private extension build definition:
  `mega_attention_build.py`
- Python routing and public advanced arguments:
  `flash_mla/mega_attention_interface.py` and
  `flash_mla/flash_mla_interface.py`
