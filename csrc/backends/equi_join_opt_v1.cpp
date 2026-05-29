#include "equi_join_constants.hpp"
#include "ops.hpp"

namespace turbogator {

void equi_join_opt_v1(const float* a, const float* b, const float* ref, float* out, size_t n, size_t ref_group) {
    const float* cur_ref = ref;
    size_t ref_left      = ref_group;
    for (size_t batch = 0; batch < n; batch++) {
        const float* cur_a = a + (batch * 16);
        const float* cur_b = b + (batch * 16);
        float* cur_out     = out + (batch * 16);

        for (int i = 0; i < 16; i++)
            cur_out[i] = 0.0f;

        for (size_t idx = 0; idx < kJoinKernelEntryCount; idx++) {
            const SparseEntry& e = kJoinKernelEntries[idx];
            cur_out[e.i] += e.v * cur_a[e.j] * cur_b[e.k];
        }

        if (ref != nullptr) {
            float scale = cur_ref[14];

            for (int i = 0; i < 16; i++)
                cur_out[i] *= scale;

            if (--ref_left == 0) {
                cur_ref += 16;
                ref_left = ref_group;
            }
        }
    }
}

}  // namespace turbogator
