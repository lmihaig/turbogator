#include "ops.hpp"

namespace tg {

void scaler_gated_gelu_baseline(const float* x, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = x[i];
    }
}

}  // namespace tg
