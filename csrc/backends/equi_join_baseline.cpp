#include "ops.hpp"

namespace turbogator {

void equi_join_baseline(const float* a, const float* b, const float* ref, float* out, size_t n) {
    (void)ref;
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

}  // namespace turbogator
