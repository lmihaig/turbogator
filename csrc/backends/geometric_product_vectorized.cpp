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

__attribute__((always_inline)) static inline void transpose_8x8(const __m256 in[8], __m256 out[8]) {
    __m256 t0 = _mm256_unpacklo_ps(in[0], in[1]);
    __m256 t1 = _mm256_unpackhi_ps(in[0], in[1]);
    __m256 t2 = _mm256_unpacklo_ps(in[2], in[3]);
    __m256 t3 = _mm256_unpackhi_ps(in[2], in[3]);
    __m256 t4 = _mm256_unpacklo_ps(in[4], in[5]);
    __m256 t5 = _mm256_unpackhi_ps(in[4], in[5]);
    __m256 t6 = _mm256_unpacklo_ps(in[6], in[7]);
    __m256 t7 = _mm256_unpackhi_ps(in[6], in[7]);

    __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

    out[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
    out[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
    out[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
    out[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
    out[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
    out[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
    out[6] = _mm256_permute2f128_ps(s2, s6, 0x31);
    out[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
}

__attribute__((always_inline)) static inline void transpose_8x16(const float* __restrict__ src,
                                                                 float* __restrict__ dst) {
    __m256 lo[8], hi[8], olo[8], ohi[8];

#pragma GCC unroll 8
    for (int r = 0; r < 8; ++r) {
        lo[r] = _mm256_loadu_ps(src + r * 16);
        hi[r] = _mm256_loadu_ps(src + r * 16 + 8);
    }

    transpose_8x8(lo, olo);
    transpose_8x8(hi, ohi);

#pragma GCC unroll 8
    for (int c = 0; c < 8; ++c) {
        _mm256_storeu_ps(dst + c * 8, olo[c]);
        _mm256_storeu_ps(dst + (c + 8) * 8, ohi[c]);
    }
}

__attribute__((always_inline)) static inline void transpose_16x8(const float* __restrict__ src,
                                                                 float* __restrict__ dst) {
    __m256 top[8], bot[8], otop[8], obot[8];

#pragma GCC unroll 8
    for (int r = 0; r < 8; ++r) {
        top[r] = _mm256_loadu_ps(src + r * 8);
        bot[r] = _mm256_loadu_ps(src + (r + 8) * 8);
    }

    transpose_8x8(top, otop);
    transpose_8x8(bot, obot);

#pragma GCC unroll 8
    for (int lane = 0; lane < 8; ++lane) {
        _mm256_storeu_ps(dst + lane * 16, otop[lane]);
        _mm256_storeu_ps(dst + lane * 16 + 8, obot[lane]);
    }
}

__attribute__((always_inline)) static inline void gp_avx2_8batch(const float* __restrict__ a_T,
                                                                 const float* __restrict__ b_T,
                                                                 float* __restrict__ out_T) {
    __m256 acc[16];

#pragma GCC unroll 16
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

#pragma GCC unroll 16
    for (int i = 0; i < 16; ++i)
        _mm256_storeu_ps(out_T + i * 8, acc[i]);
}

__attribute__((always_inline)) static inline void gp_scalar_one(const float* __restrict__ a,
                                                                const float* __restrict__ b,
                                                                float* __restrict__ out) {
#pragma GCC unroll 16
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
        transpose_8x16(ab, a_T);
        transpose_8x16(bb, b_T);
        gp_avx2_8batch(a_T, b_T, out_T);
        transpose_16x8(out_T, ob);
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