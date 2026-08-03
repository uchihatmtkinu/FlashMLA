#include "../phase1.h"
#include "../phase1.cuh"

namespace mega_attention::stock_fused {

template void run_fwd_for_small_topk_phase1_kernel<SparseAttnFwdMode::DecodeWithSplitKV, 512>(const SparseAttnDecodeParams& params);
template void run_fwd_for_small_topk_phase1_kernel<SparseAttnFwdMode::DecodeReady, 512>(const SparseAttnDecodeParams& params);

}
