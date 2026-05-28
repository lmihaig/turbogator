#include "ops.hpp"
#include "equi_join_constants.hpp"

#include <immintrin.h>
#include <stdexcept>

/*
// actually this math can be ignored, we can just hardcode the kernel
x = [a, b, c, d, ... n, o, p]
equi_dual(x)
    perm = [15, 14, 13, ... 0]
    sign = [1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1]
    return x = [p, -o, n, ... d, -c, b, a]

outer_product(x, y)
    op_basis = [16x16x16 sparse]
    for i in 0..15:
       for j in 0..15:
          for k in 0..15:
             out[i] += op_basis[i][j][k] * x[j] * y[k]

compute_join_kernel() ...
equi_join(x,)
    kernel = [16x16x16]
    for i in 0..15:
       for j in 0..15:
          for k in 0..15:
                out[i] += kernel[i][j][k] * x[j] * y[k]

    scale by ref[14]
*/

namespace turbogator
{

    // incredibly sparse matrix
    // 16x16x16 = 4096 floats
    // 81 nonzero
    // 98% sparsity

    // IMPORTANT: see random/join_kernel.txt
    struct JoinKernel
    {
        // OPTIMISATION TODO:
        // this is a constant kernel, can be hardcoded using one of the sparse formats
        // it's also just -1,0,1
        float data[16][16][16] = {};

        JoinKernel()
        {
            for (int i = 0; i < 16; i++)
            {
                for (int j = 0; j < 16; j++)
                {
                    float x[16] = {0};
                    float y[16] = {0};
                    x[i] = 1.0f;
                    y[j] = 1.0f;

                    float dual_x[16];
                    float dual_y[16];
                    float prod[16];
                    float final_out[16];

                    equi_dual(x, dual_x);
                    equi_dual(y, dual_y);
                    // outer_prod(dual_x, dual_y, prod);
                    outer_product_hardcoded(dual_x, dual_y, prod);
                    equi_dual(prod, final_out);

                    for (int k = 0; k < 16; k++)
                        data[k][i][j] = final_out[k];
                }
            }
        }

        inline void equi_dual(const float *in, float *out)
        {
            const int perm[16] = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
            const float sign[16] = {1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1};

            for (int i = 0; i < 16; i++)
                out[i] = sign[i] * in[perm[i]];
        }

        inline void outer_prod(const float *x, const float *y, float *out)
        {
            // op bilinear basis, see random/
            float OP_BASIS[16][16][16] = {};
            // i did this to hide the code block lol
            if (1)
            {

                OP_BASIS[0][0][0] = 1.0f;
                OP_BASIS[1][0][1] = 1.0f;
                OP_BASIS[1][1][0] = 1.0f;
                OP_BASIS[2][0][2] = 1.0f;
                OP_BASIS[2][2][0] = 1.0f;
                OP_BASIS[3][0][3] = 1.0f;
                OP_BASIS[3][3][0] = 1.0f;
                OP_BASIS[4][0][4] = 1.0f;
                OP_BASIS[4][4][0] = 1.0f;
                OP_BASIS[5][0][5] = 1.0f;
                OP_BASIS[5][1][2] = 1.0f;
                OP_BASIS[5][2][1] = -1.0f;
                OP_BASIS[5][5][0] = 1.0f;
                OP_BASIS[6][0][6] = 1.0f;
                OP_BASIS[6][1][3] = 1.0f;
                OP_BASIS[6][3][1] = -1.0f;
                OP_BASIS[6][6][0] = 1.0f;
                OP_BASIS[7][0][7] = 1.0f;
                OP_BASIS[7][1][4] = 1.0f;
                OP_BASIS[7][4][1] = -1.0f;
                OP_BASIS[7][7][0] = 1.0f;
                OP_BASIS[8][0][8] = 1.0f;
                OP_BASIS[8][2][3] = 1.0f;
                OP_BASIS[8][3][2] = -1.0f;
                OP_BASIS[8][8][0] = 1.0f;
                OP_BASIS[9][0][9] = 1.0f;
                OP_BASIS[9][2][4] = 1.0f;
                OP_BASIS[9][4][2] = -1.0f;
                OP_BASIS[9][9][0] = 1.0f;
                OP_BASIS[10][0][10] = 1.0f;
                OP_BASIS[10][3][4] = 1.0f;
                OP_BASIS[10][4][3] = -1.0f;
                OP_BASIS[10][10][0] = 1.0f;
                OP_BASIS[11][0][11] = 1.0f;
                OP_BASIS[11][1][8] = 1.0f;
                OP_BASIS[11][2][6] = -1.0f;
                OP_BASIS[11][3][5] = 1.0f;
                OP_BASIS[11][5][3] = 1.0f;
                OP_BASIS[11][6][2] = -1.0f;
                OP_BASIS[11][8][1] = 1.0f;
                OP_BASIS[11][11][0] = 1.0f;
                OP_BASIS[12][0][12] = 1.0f;
                OP_BASIS[12][1][9] = 1.0f;
                OP_BASIS[12][2][7] = -1.0f;
                OP_BASIS[12][4][5] = 1.0f;
                OP_BASIS[12][5][4] = 1.0f;
                OP_BASIS[12][7][2] = -1.0f;
                OP_BASIS[12][9][1] = 1.0f;
                OP_BASIS[12][12][0] = 1.0f;
                OP_BASIS[13][0][13] = 1.0f;
                OP_BASIS[13][1][10] = 1.0f;
                OP_BASIS[13][3][7] = -1.0f;
                OP_BASIS[13][4][6] = 1.0f;
                OP_BASIS[13][6][4] = 1.0f;
                OP_BASIS[13][7][3] = -1.0f;
                OP_BASIS[13][10][1] = 1.0f;
                OP_BASIS[13][13][0] = 1.0f;
                OP_BASIS[14][0][14] = 1.0f;
                OP_BASIS[14][2][10] = 1.0f;
                OP_BASIS[14][3][9] = -1.0f;
                OP_BASIS[14][4][8] = 1.0f;
                OP_BASIS[14][8][4] = 1.0f;
                OP_BASIS[14][9][3] = -1.0f;
                OP_BASIS[14][10][2] = 1.0f;
                OP_BASIS[14][14][0] = 1.0f;
                OP_BASIS[15][0][15] = 1.0f;
                OP_BASIS[15][1][14] = 1.0f;
                OP_BASIS[15][2][13] = -1.0f;
                OP_BASIS[15][3][12] = 1.0f;
                OP_BASIS[15][4][11] = -1.0f;
                OP_BASIS[15][5][10] = 1.0f;
                OP_BASIS[15][6][9] = -1.0f;
                OP_BASIS[15][7][8] = 1.0f;
                OP_BASIS[15][8][7] = 1.0f;
                OP_BASIS[15][9][6] = -1.0f;
                OP_BASIS[15][10][5] = 1.0f;
                OP_BASIS[15][11][4] = 1.0f;
                OP_BASIS[15][12][3] = -1.0f;
                OP_BASIS[15][13][2] = 1.0f;
                OP_BASIS[15][14][1] = -1.0f;
                OP_BASIS[15][15][0] = 1.0f;
            }

            for (int i = 0; i < 16; i++)
                out[i] = 0.0f;

            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++)
                    for (int k = 0; k < 16; k++)
                        out[i] += OP_BASIS[i][j][k] * x[j] * y[k];
        }

        inline void outer_product_hardcoded(const float* x, const float* y, float* out)
        {
            for (int i = 0; i < 16; i++)
                out[i] = 0.0f;

            for (size_t idx = 0; idx < kOpBasisEntryCount; idx++)
            {
                const SparseEntry& e = kOpBasisEntries[idx];
                out[e.i] += e.v * x[e.j] * y[e.k];
            }
        }
    };

    inline static const JoinKernel KERNEL;

#if defined(__AVX2__)
    struct JoinKernelByI
    {
        int counts[16];
        int j_idx[16][16];
        int k_idx[16][16];
        float v[16][16];

        JoinKernelByI()
        {
            for (int i = 0; i < 16; ++i)
            {
                counts[i] = 0;
                for (int j = 0; j < 16; ++j)
                {
                    j_idx[i][j] = 0;
                    k_idx[i][j] = 0;
                    v[i][j] = 0.0f;
                }
            }

            for (size_t idx = 0; idx < kJoinKernelEntryCount; ++idx)
            {
                const SparseEntry& e = kJoinKernelEntries[idx];
                int pos = counts[e.i]++;
                j_idx[e.i][pos] = static_cast<int>(e.j);
                k_idx[e.i][pos] = static_cast<int>(e.k);
                v[e.i][pos] = e.v;
            }
        }
    };

    inline static const JoinKernelByI KERNEL_BY_I;

    static inline float hsum256_ps(__m256 v)
    {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 sum4 = _mm_add_ps(lo, hi);
        __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
        __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 0x1));
        return _mm_cvtss_f32(sum1);
    }
#endif
    // this breaks if D != 16 !!!!
    void equi_join_baseline(const float *a, const float *b, const float *ref, float *out, size_t n)
    {
        // ret = torch.einsum("ijk, ...j, ...k -> ...i", kernel, x, y)
        // if reference is not None:
        //     ret *= reference[..., [14]]
        // return ret

        for (size_t batch = 0; batch < n; batch++)
        {
            // curr multivector
            const float *cur_a = a + (batch * 16);
            const float *cur_b = b + (batch * 16);
            float *cur_out = out + (batch * 16);

            for (int i = 0; i < 16; i++)
                cur_out[i] = 0.0f;

            // OPTIMISATION TODO:
            // actual potentials for optimisations HERE
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++)
                    for (int k = 0; k < 16; k++)
                        cur_out[i] += KERNEL.data[i][j][k] * cur_a[j] * cur_b[k];

            // scale by ref?
            if (ref != nullptr)
            {
                const float *cur_ref = ref + (batch * 16);
                float scale = cur_ref[14];

                for (int i = 0; i < 16; i++)
                    cur_out[i] *= scale;
            }
        }
    }

    void equi_join_optimized_hardcoded(const float* a, const float* b, const float* ref, float* out, size_t n)
    {
        for (size_t batch = 0; batch < n; batch++)
        {
            const float* cur_a = a + (batch * 16);
            const float* cur_b = b + (batch * 16);
            float* cur_out = out + (batch * 16);

            for (int i = 0; i < 16; i++)
                cur_out[i] = 0.0f;

            for (size_t idx = 0; idx < kJoinKernelEntryCount; idx++)
            {
                const SparseEntry& e = kJoinKernelEntries[idx];
                cur_out[e.i] += e.v * cur_a[e.j] * cur_b[e.k];
            }

            if (ref != nullptr)
            {
                const float* cur_ref = ref + (batch * 16);
                float scale = cur_ref[14];

                for (int i = 0; i < 16; i++)
                    cur_out[i] *= scale;
            }
        }
    }

    void equi_join_optimized_avx2(const float* a, const float* b, const float* ref, float* out, size_t n)
    {
#if defined(__AVX2__)
        for (size_t batch = 0; batch < n; ++batch)
        {
            const float* cur_a = a + (batch * 16);
            const float* cur_b = b + (batch * 16);
            float* cur_out = out + (batch * 16);

            for (int i = 0; i < 16; ++i)
            {
                const int count = KERNEL_BY_I.counts[i];
                const int* j_ptr = KERNEL_BY_I.j_idx[i];
                const int* k_ptr = KERNEL_BY_I.k_idx[i];
                const float* v_ptr = KERNEL_BY_I.v[i];

                __m256 acc = _mm256_setzero_ps();
                int idx = 0;

                for (; idx + 8 <= count; idx += 8)
                {
                    __m256i j_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(j_ptr + idx));
                    __m256i k_idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k_ptr + idx));
                    __m256 a_vals = _mm256_i32gather_ps(cur_a, j_idx, 4);
                    __m256 b_vals = _mm256_i32gather_ps(cur_b, k_idx, 4);
                    __m256 v_vals = _mm256_loadu_ps(v_ptr + idx);
#if defined(__FMA__)
                    __m256 b_signed = _mm256_mul_ps(b_vals, v_vals);
                    acc = _mm256_fmadd_ps(a_vals, b_signed, acc);
#else
                    __m256 prod = _mm256_mul_ps(_mm256_mul_ps(a_vals, b_vals), v_vals);
                    acc = _mm256_add_ps(acc, prod);
#endif
                }

                float sum = hsum256_ps(acc);
                for (; idx < count; ++idx)
                    sum += v_ptr[idx] * cur_a[j_ptr[idx]] * cur_b[k_ptr[idx]];

                cur_out[i] = sum;
            }

            if (ref != nullptr)
            {
                const float* cur_ref = ref + (batch * 16);
                float scale = cur_ref[14];
                __m256 s = _mm256_set1_ps(scale);
                _mm256_storeu_ps(cur_out, _mm256_mul_ps(_mm256_loadu_ps(cur_out), s));
                _mm256_storeu_ps(cur_out + 8, _mm256_mul_ps(_mm256_loadu_ps(cur_out + 8), s));
            }
        }
#else
    throw std::runtime_error("equi_join_optimized_avx2 requires AVX2 support");
#endif
    }

    void equi_join_optimized_sparse(const float *a, const float *b, const float *ref, float *out, size_t n)
    {
        struct SparseKernel
        {
            struct Entry
            {
                int i;
                int j;
                int k;
                float v;
            };

            int count;
            Entry entries[4096];

            SparseKernel() : count(0)
            {
                for (int i = 0; i < 16; i++)
                {
                    for (int j = 0; j < 16; j++)
                    {
                        for (int k = 0; k < 16; k++)
                        {
                            float v = KERNEL.data[i][j][k];
                            if (v != 0.0f)
                            {
                                entries[count++] = {i, j, k, v};
                            }
                        }
                    }
                }
            }
        };

        static const SparseKernel sk;

        for (size_t batch = 0; batch < n; batch++)
        {
            const float *cur_a = a + (batch * 16);
            const float *cur_b = b + (batch * 16);
            float *cur_out = out + (batch * 16);

            for (int i = 0; i < 16; i++)
                cur_out[i] = 0.0f;

            for (int idx = 0; idx < sk.count; idx++)
            {
                const auto &e = sk.entries[idx];
                cur_out[e.i] += e.v * cur_a[e.j] * cur_b[e.k];
            }

            if (ref != nullptr)
            {
                const float *cur_ref = ref + (batch * 16);
                float scale = cur_ref[14];

                for (int i = 0; i < 16; i++)
                    cur_out[i] *= scale;
            }
        }
    }

    void equi_join_optimized_precompute_ab(const float *a, const float *b, const float *ref, float *out, size_t n)
    {
        for (size_t batch = 0; batch < n; batch++)
        {
            const float *cur_a = a + (batch * 16);
            const float *cur_b = b + (batch * 16);
            float *cur_out = out + (batch * 16);

            float ab[16][16];
            for (int j = 0; j < 16; j++)
            {
                float aj = cur_a[j];
                for (int k = 0; k < 16; k++)
                    ab[j][k] = aj * cur_b[k];
            }

            for (int i = 0; i < 16; i++)
                cur_out[i] = 0.0f;

            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++)
                    for (int k = 0; k < 16; k++)
                        cur_out[i] += KERNEL.data[i][j][k] * ab[j][k];

            if (ref != nullptr)
            {
                const float *cur_ref = ref + (batch * 16);
                float scale = cur_ref[14];

                for (int i = 0; i < 16; i++)
                    cur_out[i] *= scale;
            }
        }
    }

    void equi_join_optimized_unroll_k(const float *a, const float *b, const float *ref, float *out, size_t n)
    {
        for (size_t batch = 0; batch < n; batch++)
        {
            const float *cur_a = a + (batch * 16);
            const float *cur_b = b + (batch * 16);
            float *cur_out = out + (batch * 16);

            for (int i = 0; i < 16; i++)
                cur_out[i] = 0.0f;

            for (int i = 0; i < 16; i++)
            {
                for (int j = 0; j < 16; j++)
                {
                    float aj = cur_a[j];
                    cur_out[i] += KERNEL.data[i][j][0] * aj * cur_b[0];
                    cur_out[i] += KERNEL.data[i][j][1] * aj * cur_b[1];
                    cur_out[i] += KERNEL.data[i][j][2] * aj * cur_b[2];
                    cur_out[i] += KERNEL.data[i][j][3] * aj * cur_b[3];
                    cur_out[i] += KERNEL.data[i][j][4] * aj * cur_b[4];
                    cur_out[i] += KERNEL.data[i][j][5] * aj * cur_b[5];
                    cur_out[i] += KERNEL.data[i][j][6] * aj * cur_b[6];
                    cur_out[i] += KERNEL.data[i][j][7] * aj * cur_b[7];
                    cur_out[i] += KERNEL.data[i][j][8] * aj * cur_b[8];
                    cur_out[i] += KERNEL.data[i][j][9] * aj * cur_b[9];
                    cur_out[i] += KERNEL.data[i][j][10] * aj * cur_b[10];
                    cur_out[i] += KERNEL.data[i][j][11] * aj * cur_b[11];
                    cur_out[i] += KERNEL.data[i][j][12] * aj * cur_b[12];
                    cur_out[i] += KERNEL.data[i][j][13] * aj * cur_b[13];
                    cur_out[i] += KERNEL.data[i][j][14] * aj * cur_b[14];
                    cur_out[i] += KERNEL.data[i][j][15] * aj * cur_b[15];
                }
            }

            if (ref != nullptr)
            {
                const float *cur_ref = ref + (batch * 16);
                float scale = cur_ref[14];

                for (int i = 0; i < 16; i++)
                    cur_out[i] *= scale;
            }
        }
    }

    #ifndef __has_builtin
    #define __restrict__ __restrict
    #endif

    void equi_join_restrict_unswitch(const float* __restrict__ a, 
                                    const float* __restrict__ b, 
                                    const float* __restrict__ ref, 
                                    float* __restrict__ out, 
                                    size_t n)
    {
        if (ref != nullptr)
        {
            for (size_t batch = 0; batch < n; batch++)
            {
                const float *cur_a = a + (batch * 16);
                const float *cur_b = b + (batch * 16);
                float *cur_out = out + (batch * 16);
                const float *cur_ref = ref + (batch * 16);

                for (int i = 0; i < 16; i++) cur_out[i] = 0.0f;

                for (int i = 0; i < 16; i++)
                    for (int j = 0; j < 16; j++)
                        for (int k = 0; k < 16; k++)
                            cur_out[i] += turbogator::KERNEL.data[i][j][k] * cur_a[j] * cur_b[k];

                float scale = cur_ref[14];
                for (int i = 0; i < 16; i++)
                    cur_out[i] *= scale;
            }
        }
        else
        {
            for (size_t batch = 0; batch < n; batch++)
            {
                const float *cur_a = a + (batch * 16);
                const float *cur_b = b + (batch * 16);
                float *cur_out = out + (batch * 16);

                for (int i = 0; i < 16; i++) cur_out[i] = 0.0f;

                for (int i = 0; i < 16; i++)
                    for (int j = 0; j < 16; j++)
                        for (int k = 0; k < 16; k++)
                            cur_out[i] += turbogator::KERNEL.data[i][j][k] * cur_a[j] * cur_b[k];
            }
        }
    }

} // namespace turbogator
