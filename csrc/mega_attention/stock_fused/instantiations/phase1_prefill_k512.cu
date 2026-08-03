#include "../phase1.h"
#include "../phase1.cuh"

namespace mega_attention::stock_fused {

template void run_fwd_for_small_topk_phase1_kernel<SparseAttnFwdMode::Prefill, 512>(const SparseAttnFwdParams& params);

}
