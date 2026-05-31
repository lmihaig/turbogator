#include <immintrin.h>

#include "equi_join_constants.hpp"
#include "ops.hpp"

namespace turbogator {

struct JoinKernelByI {
    int counts[16];
    int j_idx[16][16];
    int k_idx[16][16];
    float v[16][16];

    JoinKernelByI() {
        for (int i = 0; i < 16; ++i) {
            counts[i] = 0;
            for (int j = 0; j < 16; ++j) {
                j_idx[i][j] = 0;
                k_idx[i][j] = 0;
                v[i][j]     = 0.0f;
            }
        }

        for (size_t idx = 0; idx < kJoinKernelEntryCount; ++idx) {
            const SparseEntry& e = kJoinKernelEntries[idx];
            int pos              = counts[e.i]++;
            j_idx[e.i][pos]      = static_cast<int>(e.j);
            k_idx[e.i][pos]      = static_cast<int>(e.k);
            v[e.i][pos]          = e.v;
        }
    }
};

inline static const JoinKernelByI KERNEL_BY_I;

__attribute__((always_inline)) static inline float hsum256_ps(__m256 v) {
    __m128 lo   = _mm256_castps256_ps128(v);
    __m128 hi   = _mm256_extractf128_ps(v, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 0x1));
    return _mm_cvtss_f32(sum1);
}

void equi_join_vectorized_out(const float* __restrict__ a,
                              const float* __restrict__ b,
                              const float* __restrict__ ref,
                              float* __restrict__ out,
                              size_t n,
                              size_t ref_group,
                              size_t block_size,
                              size_t outer_stride_a,
                              size_t outer_stride_b,
                              size_t outer_stride_out) {
    const float* cur_ref    = ref;
    size_t ref_left         = ref_group;
    const size_t n_outer    = (block_size > 0) ? n / block_size : 1;
    const size_t inner_size = (block_size > 0) ? block_size : n;

    const size_t row_out = outer_stride_out ? outer_stride_out : inner_size * 16;

    for (size_t outer = 0; outer < n_outer; ++outer) {
        const float* a_base = (block_size > 0) ? a + outer * outer_stride_a : a;
        const float* b_base = (block_size > 0) ? b + outer * outer_stride_b : b;
        float* o_base       = out + outer * row_out;

        for (size_t batch = 0; batch < inner_size; ++batch) {
            const float* cur_a = a_base + batch * 16;
            const float* cur_b = b_base + batch * 16;
            float* cur_out     = o_base + batch * 16;

            for (int i = 0; i < 16; ++i) {
                const int count    = KERNEL_BY_I.counts[i];
                const int* j_ptr   = KERNEL_BY_I.j_idx[i];
                const int* k_ptr   = KERNEL_BY_I.k_idx[i];
                const float* v_ptr = KERNEL_BY_I.v[i];

                __m256 acc = _mm256_setzero_ps();
                int idx    = 0;

                for (; idx + 8 <= count; idx += 8) {
                    __m256i j_idx   = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(j_ptr + idx));
                    __m256i k_idx   = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k_ptr + idx));
                    __m256 a_vals   = _mm256_i32gather_ps(cur_a, j_idx, 4);
                    __m256 b_vals   = _mm256_i32gather_ps(cur_b, k_idx, 4);
                    __m256 v_vals   = _mm256_loadu_ps(v_ptr + idx);
                    __m256 b_signed = _mm256_mul_ps(b_vals, v_vals);
                    acc             = _mm256_fmadd_ps(a_vals, b_signed, acc);
                }

                float sum = hsum256_ps(acc);
                for (; idx < count; ++idx)
                    sum += v_ptr[idx] * cur_a[j_ptr[idx]] * cur_b[k_ptr[idx]];

                cur_out[i] = sum;
            }

            if (ref != nullptr) {
                float scale = cur_ref[14];
                __m256 s    = _mm256_set1_ps(scale);
                _mm256_storeu_ps(cur_out, _mm256_mul_ps(_mm256_loadu_ps(cur_out), s));
                _mm256_storeu_ps(cur_out + 8, _mm256_mul_ps(_mm256_loadu_ps(cur_out + 8), s));

                if (--ref_left == 0) {
                    cur_ref += 16;
                    ref_left = ref_group;
                }
            }
        }
    }
}

void equi_join_vectorized(const float* __restrict__ a,
                          const float* __restrict__ b,
                          const float* __restrict__ ref,
                          float* __restrict__ out,
                          size_t n,
                          size_t ref_group,
                          size_t block_size,
                          size_t outer_stride_a,
                          size_t outer_stride_b) {
    equi_join_vectorized_out(a, b, ref, out, n, ref_group, block_size, outer_stride_a, outer_stride_b, 0);
}

}  // namespace turbogator