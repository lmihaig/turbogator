#include "ops.hpp"

namespace turbogator {

void equi_rms_norm_baseline(const float* x, const float* weight, float* out, size_t n) {
    (void)x;
    (void)weight;
    for (size_t i = 0; i < n; ++i) {
        out[i] = 0.0f;
    }
}

}  // namespace turbogator
