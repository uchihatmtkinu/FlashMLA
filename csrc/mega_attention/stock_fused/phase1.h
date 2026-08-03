#pragma once

#include "../params.h"

namespace mega_attention::stock_fused {

template<SparseAttnFwdMode FWD_MODE, int D_QK>
void run_fwd_for_small_topk_phase1_kernel(const SparseFwdArgT<FWD_MODE>& params);

}
