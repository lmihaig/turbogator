#include "ops.hpp"

namespace tg {

void geometric_product_vectorized(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

}  // namespace tg
