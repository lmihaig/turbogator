#include <immintrin.h>

#include <vector>

#include "ops.hpp"

namespace turbogator {

namespace {
struct GPEntry {
    int i, j, k, val;
};

static constexpr GPEntry data[] = {
    { 0,  0,  0,  1},
    { 0,  2,  2,  1},
    { 0,  3,  3,  1},
    { 0,  4,  4,  1},
    { 0,  8,  8, -1},
    { 0,  9,  9, -1},
    { 0, 10, 10, -1},
    { 0, 14, 14, -1},
    { 1,  0,  1,  1},
    { 1,  1,  0,  1},
    { 1,  2,  5, -1},
    { 1,  3,  6, -1},
    { 1,  4,  7, -1},
    { 1,  5,  2,  1},
    { 1,  6,  3,  1},
    { 1,  7,  4,  1},
    { 1,  8, 11, -1},
    { 1,  9, 12, -1},
    { 1, 10, 13, -1},
    { 1, 11,  8, -1},
    { 1, 12,  9, -1},
    { 1, 13, 10, -1},
    { 1, 14, 15,  1},
    { 1, 15, 14, -1},
    { 2,  0,  2,  1},
    { 2,  2,  0,  1},
    { 2,  3,  8, -1},
    { 2,  4,  9, -1},
    { 2,  8,  3,  1},
    { 2,  9,  4,  1},
    { 2, 10, 14, -1},
    { 2, 14, 10, -1},
    { 3,  0,  3,  1},
    { 3,  2,  8,  1},
    { 3,  3,  0,  1},
    { 3,  4, 10, -1},
    { 3,  8,  2, -1},
    { 3,  9, 14,  1},
    { 3, 10,  4,  1},
    { 3, 14,  9,  1},
    { 4,  0,  4,  1},
    { 4,  2,  9,  1},
    { 4,  3, 10,  1},
    { 4,  4,  0,  1},
    { 4,  8, 14, -1},
    { 4,  9,  2, -1},
    { 4, 10,  3, -1},
    { 4, 14,  8, -1},
    { 5,  0,  5,  1},
    { 5,  1,  2,  1},
    { 5,  2,  1, -1},
    { 5,  3, 11,  1},
    { 5,  4, 12,  1},
    { 5,  5,  0,  1},
    { 5,  6,  8, -1},
    { 5,  7,  9, -1},
    { 5,  8,  6,  1},
    { 5,  9,  7,  1},
    { 5, 10, 15, -1},
    { 5, 11,  3,  1},
    { 5, 12,  4,  1},
    { 5, 13, 14, -1},
    { 5, 14, 13,  1},
    { 5, 15, 10, -1},
    { 6,  0,  6,  1},
    { 6,  1,  3,  1},
    { 6,  2, 11, -1},
    { 6,  3,  1, -1},
    { 6,  4, 13,  1},
    { 6,  5,  8,  1},
    { 6,  6,  0,  1},
    { 6,  7, 10, -1},
    { 6,  8,  5, -1},
    { 6,  9, 15,  1},
    { 6, 10,  7,  1},
    { 6, 11,  2, -1},
    { 6, 12, 14,  1},
    { 6, 13,  4,  1},
    { 6, 14, 12, -1},
    { 6, 15,  9,  1},
    { 7,  0,  7,  1},
    { 7,  1,  4,  1},
    { 7,  2, 12, -1},
    { 7,  3, 13, -1},
    { 7,  4,  1, -1},
    { 7,  5,  9,  1},
    { 7,  6, 10,  1},
    { 7,  7,  0,  1},
    { 7,  8, 15, -1},
    { 7,  9,  5, -1},
    { 7, 10,  6, -1},
    { 7, 11, 14, -1},
    { 7, 12,  2, -1},
    { 7, 13,  3, -1},
    { 7, 14, 11,  1},
    { 7, 15,  8, -1},
    { 8,  0,  8,  1},
    { 8,  2,  3,  1},
    { 8,  3,  2, -1},
    { 8,  4, 14,  1},
    { 8,  8,  0,  1},
    { 8,  9, 10, -1},
    { 8, 10,  9,  1},
    { 8, 14,  4,  1},
    { 9,  0,  9,  1},
    { 9,  2,  4,  1},
    { 9,  3, 14, -1},
    { 9,  4,  2, -1},
    { 9,  8, 10,  1},
    { 9,  9,  0,  1},
    { 9, 10,  8, -1},
    { 9, 14,  3, -1},
    {10,  0, 10,  1},
    {10,  2, 14,  1},
    {10,  3,  4,  1},
    {10,  4,  3, -1},
    {10,  8,  9, -1},
    {10,  9,  8,  1},
    {10, 10,  0,  1},
    {10, 14,  2,  1},
    {11,  0, 11,  1},
    {11,  1,  8,  1},
    {11,  2,  6, -1},
    {11,  3,  5,  1},
    {11,  4, 15, -1},
    {11,  5,  3,  1},
    {11,  6,  2, -1},
    {11,  7, 14,  1},
    {11,  8,  1,  1},
    {11,  9, 13, -1},
    {11, 10, 12,  1},
    {11, 11,  0,  1},
    {11, 12, 10, -1},
    {11, 13,  9,  1},
    {11, 14,  7, -1},
    {11, 15,  4,  1},
    {12,  0, 12,  1},
    {12,  1,  9,  1},
    {12,  2,  7, -1},
    {12,  3, 15,  1},
    {12,  4,  5,  1},
    {12,  5,  4,  1},
    {12,  6, 14, -1},
    {12,  7,  2, -1},
    {12,  8, 13,  1},
    {12,  9,  1,  1},
    {12, 10, 11, -1},
    {12, 11, 10,  1},
    {12, 12,  0,  1},
    {12, 13,  8, -1},
    {12, 14,  6,  1},
    {12, 15,  3, -1},
    {13,  0, 13,  1},
    {13,  1, 10,  1},
    {13,  2, 15, -1},
    {13,  3,  7, -1},
    {13,  4,  6,  1},
    {13,  5, 14,  1},
    {13,  6,  4,  1},
    {13,  7,  3, -1},
    {13,  8, 12, -1},
    {13,  9, 11,  1},
    {13, 10,  1,  1},
    {13, 11,  9, -1},
    {13, 12,  8,  1},
    {13, 13,  0,  1},
    {13, 14,  5, -1},
    {13, 15,  2,  1},
    {14,  0, 14,  1},
    {14,  2, 10,  1},
    {14,  3,  9, -1},
    {14,  4,  8,  1},
    {14,  8,  4,  1},
    {14,  9,  3, -1},
    {14, 10,  2,  1},
    {14, 14,  0,  1},
    {15,  0, 15,  1},
    {15,  1, 14,  1},
    {15,  2, 13, -1},
    {15,  3, 12,  1},
    {15,  4, 11, -1},
    {15,  5, 10,  1},
    {15,  6,  9, -1},
    {15,  7,  8,  1},
    {15,  8,  7,  1},
    {15,  9,  6, -1},
    {15, 10,  5,  1},
    {15, 11,  4,  1},
    {15, 12,  3, -1},
    {15, 13,  2,  1},
    {15, 14,  1, -1},
    {15, 15,  0,  1},
};

struct GPSplit {
    std::vector<GPEntry> pos;
    std::vector<GPEntry> neg;
    GPSplit() {
        for (const auto& en : data) {
            if (en.val > 0)
                pos.push_back(en);
            else
                neg.push_back(en);
        }
    }
};
static const GPSplit split;

template <int R, int C>
void transpose(const float* src, float* dst) {
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
            dst[c * R + r] = src[r * C + c];
}

__attribute__((always_inline)) static inline void gp_avx2_8batch(const float* __restrict__ a_T,
                                                                 const float* __restrict__ b_T,
                                                                 float* __restrict__ out_T) {
    __m256 acc[16];
    for (int i = 0; i < 16; ++i)
        acc[i] = _mm256_setzero_ps();

    for (const auto& en : split.pos) {
        __m256 va = _mm256_loadu_ps(a_T + en.j * 8);
        __m256 vb = _mm256_loadu_ps(b_T + en.k * 8);
        acc[en.i] = _mm256_fmadd_ps(va, vb, acc[en.i]);
    }
    for (const auto& en : split.neg) {
        __m256 va = _mm256_loadu_ps(a_T + en.j * 8);
        __m256 vb = _mm256_loadu_ps(b_T + en.k * 8);
        acc[en.i] = _mm256_fnmadd_ps(va, vb, acc[en.i]);
    }

    for (int i = 0; i < 16; ++i)
        _mm256_storeu_ps(out_T + i * 8, acc[i]);
}

__attribute__((always_inline)) static inline void gp_scalar_one(const float* __restrict__ a,
                                                                const float* __restrict__ b,
                                                                float* __restrict__ out) {
    for (int i = 0; i < 16; ++i)
        out[i] = 0.0f;
    for (const auto& en : data) {
        if (en.val > 0)
            out[en.i] += a[en.j] * b[en.k];
        else
            out[en.i] -= a[en.j] * b[en.k];
    }
}

}  // namespace

static void gp_block(const float* __restrict__ a_base,
                     const float* __restrict__ b_base,
                     float* __restrict__ o_base,
                     size_t mv_count) {
    constexpr size_t BLOCK = 8;
    alignas(32) float a_T[16 * BLOCK];
    alignas(32) float b_T[16 * BLOCK];
    alignas(32) float out_T[16 * BLOCK];
    const size_t full = mv_count / BLOCK;
    for (size_t bk = 0; bk < full; ++bk) {
        const float* ab = a_base + bk * BLOCK * 16;
        const float* bb = b_base + bk * BLOCK * 16;
        float* ob       = o_base + bk * BLOCK * 16;
        transpose<8, 16>(ab, a_T);
        transpose<8, 16>(bb, b_T);
        gp_avx2_8batch(a_T, b_T, out_T);
        transpose<16, 8>(out_T, ob);
    }
    for (size_t i = full * BLOCK; i < mv_count; ++i)
        gp_scalar_one(a_base + i * 16, b_base + i * 16, o_base + i * 16);
}

void geometric_product_vectorized(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ out,
                                  size_t n,
                                  size_t block_size,
                                  size_t outer_stride_a,
                                  size_t outer_stride_b) {
    if (block_size == 0) {
        // contiguous path
        if (n % 8 != 0)
            __builtin_unreachable();
        gp_block(a, b, out, n);
    } else {
        // strided path: n/block_size outer iterations
        const size_t n_outer = n / block_size;
        for (size_t outer = 0; outer < n_outer; ++outer) {
            gp_block(a + outer * outer_stride_a, b + outer * outer_stride_b, out + outer * block_size * 16, block_size);
        }
    }
}

}  // namespace turbogator
