#include "equi_join_kernel.hpp"
#include "ops.hpp"

namespace turbogator {

void equi_join_opt_v2(const float* a, const float* b, const float* ref, float* out, size_t n, size_t ref_group) {
    struct SparseKernel {
        struct Entry {
            int i;
            int j;
            int k;
            float v;
        };

        int count;
        Entry entries[4096];

        SparseKernel() : count(0) {
            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 16; j++) {
                    for (int k = 0; k < 16; k++) {
                        float v = KERNEL.data[i][j][k];
                        if (v != 0.0f) {
                            entries[count++] = {i, j, k, v};
                        }
                    }
                }
            }
        }
    };

    static const SparseKernel sk;

    const float* cur_ref = ref;
    size_t ref_left      = ref_group;
    for (size_t batch = 0; batch < n; batch++) {
        const float* cur_a = a + (batch * 16);
        const float* cur_b = b + (batch * 16);
        float* cur_out     = out + (batch * 16);

        for (int i = 0; i < 16; i++)
            cur_out[i] = 0.0f;

        for (int idx = 0; idx < sk.count; idx++) {
            const auto& e = sk.entries[idx];
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
