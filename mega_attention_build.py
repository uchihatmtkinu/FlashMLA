from pathlib import Path

from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME


def make_mega_attention_extension(
    repo_root,
    cxx_args,
    feature_args,
    arch_flags,
    nvcc_thread_args,
):
    """Build the isolated SM100 MegaAttention extension."""
    root = Path(repo_root)
    return CUDAExtension(
        name="flash_mla.mega_attention_cuda",
        sources=[
            "csrc/mega_attention/api.cpp",
            "csrc/mega_attention/stock_fused/instantiations/phase1_prefill_k512.cu",
            "csrc/mega_attention/stock_fused/instantiations/phase1_decode_k512.cu",
            "csrc/mega_attention/mega_fwd/sparse_attention.cu",
        ],
        extra_compile_args={
            "cxx": cxx_args + feature_args,
            "nvcc": [
                "-O3",
                "-std=c++20",
                "-DNDEBUG",
                "-D_USE_MATH_DEFINES",
                "-Wno-deprecated-declarations",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-local-memory-usage,--warn-on-double-precision-use",
                "-lineinfo",
                "--source-in-ptx",
            ] + feature_args + arch_flags + nvcc_thread_args,
        },
        include_dirs=[
            Path(CUDA_HOME) / "include" / "cccl",
            root / "csrc",
            root / "csrc" / "kerutils" / "include",
            root / "csrc" / "sm90",
            root / "csrc" / "cutlass" / "include",
            root / "csrc" / "cutlass" / "tools" / "util" / "include",
        ],
    )
