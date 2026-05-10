#include "ops.hpp"

namespace tg {

void equi_geometric_attention_baseline(const float* q, const float* k, const float* v, float* out, size_t n) {
    (void)k;
    for (size_t i = 0; i < n; ++i) {
        out[i] = q[i] + v[i];
    }
}

}  // namespace tg
