#include "ops.hpp"

#include <cmath>

namespace turbogator {

// PGA inner product selects blades that don't contain e_0
static const int IP_SELECTOR[] = {0, 2, 3, 4, 8, 9, 10, 14};
static const int IP_SELECTOR_LEN = 8;

void equi_rms_norm_baseline(const float* x, const float* weight, float* out,
                            size_t batch, size_t n_channels, float eps)
{
    for (size_t b = 0; b < batch; ++b) {
        const float* x_b = x + b * n_channels * 16;

        // mean over channels of selective inner_product(x, x) from PGA
        float norm = 0.0f;
        for (size_t c = 0; c < n_channels; ++c) {
            const float* mv = x_b + c * 16;
            float ip = 0.0f;
            for (int s = 0; s < IP_SELECTOR_LEN; ++s)
                ip += mv[IP_SELECTOR[s]] * mv[IP_SELECTOR[s]];
            norm += ip;
        }
        norm /= (float)n_channels;

        float scale = 1.0f / std::sqrt(norm > eps ? norm : eps);

        float* out_b = out + b * n_channels * 16;
        for (size_t c = 0; c < n_channels; ++c) {
            float w = (weight != nullptr) ? weight[c] : 1.0f;
            float cscale = scale * w;
            const float* x_c = x_b + c * 16;
            float*      o_c  = out_b + c * 16;
            for (int d = 0; d < 16; ++d)
                o_c[d] = x_c[d] * cscale;
        }
    }
}

void equi_rms_norm_branchless_clamp(const float* x, const float* weight,
                                    float* out, size_t batch,
                                    size_t n_channels, float eps)
{
    for (size_t b = 0; b < batch; ++b) {
        const float* x_b = x + b * n_channels * 16;

        float norm = 0.0f;
        for (size_t c = 0; c < n_channels; ++c) {
            const float* mv = x_b + c * 16;
            float ip = 0.0f;
            for (int s = 0; s < IP_SELECTOR_LEN; ++s)
                ip += mv[IP_SELECTOR[s]] * mv[IP_SELECTOR[s]];
            norm += ip;
        }
        norm /= (float)n_channels;

        float clamped = std::fmax(norm, eps);
        float scale   = 1.0f / std::sqrt(clamped);

        float* out_b = out + b * n_channels * 16;
        for (size_t c = 0; c < n_channels; ++c) {
            float w = (weight != nullptr) ? weight[c] : 1.0f;
            float cscale = scale * w;
            const float* x_c = x_b + c * 16;
            float*      o_c  = out_b + c * 16;
            for (int d = 0; d < 16; ++d)
                o_c[d] = x_c[d] * cscale;
        }
    }
}

void equi_rms_norm_restrict(const float* x, const float* weight, float* out,
                            size_t batch, size_t n_channels, float eps)
{
    const float* __restrict__ x_r = x;
    const float* __restrict__ w_r = weight;
    float* __restrict__ out_r     = out;

    for (size_t b = 0; b < batch; ++b) {
        const float* x_b = x_r + b * n_channels * 16;

        float norm = 0.0f;
        for (size_t c = 0; c < n_channels; ++c) {
            const float* mv = x_b + c * 16;
            float ip = 0.0f;
            for (int s = 0; s < IP_SELECTOR_LEN; ++s)
                ip += mv[IP_SELECTOR[s]] * mv[IP_SELECTOR[s]];
            norm += ip;
        }
        norm /= (float)n_channels;

        float scale = 1.0f / std::sqrt(norm > eps ? norm : eps);

        float* out_b = out_r + b * n_channels * 16;
        for (size_t c = 0; c < n_channels; ++c) {
            float w = (w_r != nullptr) ? w_r[c] : 1.0f;
            float cscale = scale * w;
            const float* x_c = x_b + c * 16;
            float*      o_c  = out_b + c * 16;
            for (int d = 0; d < 16; ++d)
                o_c[d] = x_c[d] * cscale;
        }
    }
}

void equi_rms_norm_unrolled_selector(const float* x, const float* weight,
                                     float* out, size_t batch,
                                     size_t n_channels, float eps)
{
    for (size_t b = 0; b < batch; ++b) {
        const float* x_b = x + b * n_channels * 16;

        float norm = 0.0f;
        for (size_t c = 0; c < n_channels; ++c) {
            const float* mv = x_b + c * 16;
            float ip = mv[0] * mv[0] + mv[2] * mv[2] + mv[3] * mv[3]
                       + mv[4] * mv[4] + mv[8] * mv[8] + mv[9] * mv[9]
                       + mv[10] * mv[10] + mv[14] * mv[14];
            norm += ip;
        }
        norm /= (float)n_channels;

        float scale = 1.0f / std::sqrt(norm > eps ? norm : eps);

        float* out_b = out + b * n_channels * 16;
        for (size_t c = 0; c < n_channels; ++c) {
            float w = (weight != nullptr) ? weight[c] : 1.0f;
            float cscale = scale * w;
            const float* x_c = x_b + c * 16;
            float*      o_c  = out_b + c * 16;
            for (int d = 0; d < 16; ++d)
                o_c[d] = x_c[d] * cscale;
        }
    }
}

void equi_rms_norm_reciprocal_div(const float* x, const float* weight,
                                  float* out, size_t batch, size_t n_channels,
                                  float eps)
{
    const float inv_n_channels = 1.0f / (float)n_channels;

    for (size_t b = 0; b < batch; ++b) {
        const float* x_b = x + b * n_channels * 16;

        // mean over channels of selective inner_product(x, x) from PGA
        float norm = 0.0f;
        for (size_t c = 0; c < n_channels; ++c) {
            const float* mv = x_b + c * 16;
            float ip = 0.0f;
            for (int s = 0; s < IP_SELECTOR_LEN; ++s)
                ip += mv[IP_SELECTOR[s]] * mv[IP_SELECTOR[s]];
            norm += ip;
        }
        norm *= inv_n_channels;  // Multiplication instead of division

        float scale = 1.0f / std::sqrt(norm > eps ? norm : eps);

        float* out_b = out + b * n_channels * 16;
        for (size_t c = 0; c < n_channels; ++c) {
            float w = (weight != nullptr) ? weight[c] : 1.0f;
            float cscale = scale * w;
            const float* x_c = x_b + c * 16;
            float*      o_c  = out_b + c * 16;
            for (int d = 0; d < 16; ++d)
                o_c[d] = x_c[d] * cscale;
        }
    }
}

void equi_rms_norm_prefetch(const float* x, const float* weight, float* out,
                            size_t batch, size_t n_channels, float eps)
{
    for (size_t b = 0; b < batch; ++b) {
        const float* x_b = x + b * n_channels * 16;

        // Prefetch next batch data (read-only)
        if (b + 1 < batch) {
            const float* x_next = x + (b + 1) * n_channels * 16;
            for (size_t i = 0; i < n_channels * 16; i += 64 / sizeof(float)) {
                __builtin_prefetch(x_next + i, 0, 3);  // Prefetch for read, high locality
            }
        }

        // mean over channels of selective inner_product(x, x) from PGA
        float norm = 0.0f;
        for (size_t c = 0; c < n_channels; ++c) {
            const float* mv = x_b + c * 16;
            float ip = 0.0f;
            for (int s = 0; s < IP_SELECTOR_LEN; ++s)
                ip += mv[IP_SELECTOR[s]] * mv[IP_SELECTOR[s]];
            norm += ip;
        }
        norm /= (float)n_channels;

        float scale = 1.0f / std::sqrt(norm > eps ? norm : eps);

        float* out_b = out + b * n_channels * 16;
        for (size_t c = 0; c < n_channels; ++c) {
            float w = (weight != nullptr) ? weight[c] : 1.0f;
            float cscale = scale * w;
            const float* x_c = x_b + c * 16;
            float*      o_c  = out_b + c * 16;
            for (int d = 0; d < 16; ++d)
                o_c[d] = x_c[d] * cscale;
        }
    }
}

void equi_rms_norm_unrolled_channels_4(const float* x, const float* weight,
                                       float* out, size_t batch,
                                       size_t n_channels, float eps)
{
    for (size_t b = 0; b < batch; ++b) {
        const float* x_b = x + b * n_channels * 16;

        // mean over channels of selective inner_product(x, x) from PGA
        float norm = 0.0f;

        // Process channels in groups of 4
        size_t full_groups = n_channels / 4;
        for (size_t cg = 0; cg < full_groups; ++cg) {
            // Unroll 4 independent channel computations
            const float* mv0 = x_b + (cg * 4) * 16;
            const float* mv1 = x_b + (cg * 4 + 1) * 16;
            const float* mv2 = x_b + (cg * 4 + 2) * 16;
            const float* mv3 = x_b + (cg * 4 + 3) * 16;

            float ip0 = 0.0f, ip1 = 0.0f, ip2 = 0.0f, ip3 = 0.0f;

            for (int s = 0; s < IP_SELECTOR_LEN; ++s) {
                int idx = IP_SELECTOR[s];
                ip0 += mv0[idx] * mv0[idx];
                ip1 += mv1[idx] * mv1[idx];
                ip2 += mv2[idx] * mv2[idx];
                ip3 += mv3[idx] * mv3[idx];
            }

            norm += ip0 + ip1 + ip2 + ip3;
        }

        // Handle remainder channels
        for (size_t c = full_groups * 4; c < n_channels; ++c) {
            const float* mv = x_b + c * 16;
            float ip = 0.0f;
            for (int s = 0; s < IP_SELECTOR_LEN; ++s)
                ip += mv[IP_SELECTOR[s]] * mv[IP_SELECTOR[s]];
            norm += ip;
        }

        norm /= (float)n_channels;

        float scale = 1.0f / std::sqrt(norm > eps ? norm : eps);

        float* out_b = out + b * n_channels * 16;
        for (size_t c = 0; c < n_channels; ++c) {
            float w = (weight != nullptr) ? weight[c] : 1.0f;
            float cscale = scale * w;
            const float* x_c = x_b + c * 16;
            float*      o_c  = out_b + c * 16;
            for (int d = 0; d < 16; ++d)
                o_c[d] = x_c[d] * cscale;
        }
    }
}

void equi_rms_norm_assume_aligned(const float* x, const float* weight, float* out,
                                  size_t batch, size_t n_channels, float eps)
{
    // Inform the compiler of both strict aliasing and guaranteed 64-byte alignment
    const float* __restrict__ x_a = (const float*)__builtin_assume_aligned(x, 64);
    float* __restrict__ out_a = (float*)__builtin_assume_aligned(out, 64);

    for (size_t b = 0; b < batch; ++b) {
        const float* x_b = x_a + b * n_channels * 16;

        float norm = 0.0f;
        for (size_t c = 0; c < n_channels; ++c) {
            const float* mv = x_b + c * 16;
            float ip = 0.0f;
            for (int s = 0; s < IP_SELECTOR_LEN; ++s)
                ip += mv[IP_SELECTOR[s]] * mv[IP_SELECTOR[s]];
            norm += ip;
        }
        norm /= (float)n_channels;

        float scale = 1.0f / std::sqrt(norm > eps ? norm : eps);

        float* out_b = out_a + b * n_channels * 16;
        for (size_t c = 0; c < n_channels; ++c) {
            float w = (weight != nullptr) ? weight[c] : 1.0f;
            float cscale = scale * w;
            const float* x_c = x_b + c * 16;
            float* o_c = out_b + c * 16;
            for (int d = 0; d < 16; ++d)
                o_c[d] = x_c[d] * cscale;
        }
    }
}

void equi_rms_norm_combined(const float* x, const float* weight, float* out,
                            size_t batch, size_t n_channels, float eps)
{
    const float* __restrict__ x_a = (const float*)__builtin_assume_aligned(x, 64);
    const float* __restrict__ w_r = weight;
    float* __restrict__ out_a = (float*)__builtin_assume_aligned(out, 64);
    const float inv_n_channels = 1.0f / (float)n_channels;

    for (size_t b = 0; b < batch; ++b) {
        const float* x_b = x_a + b * n_channels * 16;

        float norm = 0.0f;
        size_t full_groups = n_channels / 4;
        for (size_t cg = 0; cg < full_groups; ++cg) {
            const float* mv0 = x_b + (cg * 4) * 16;
            const float* mv1 = x_b + (cg * 4 + 1) * 16;
            const float* mv2 = x_b + (cg * 4 + 2) * 16;
            const float* mv3 = x_b + (cg * 4 + 3) * 16;

            float ip0 = mv0[0] * mv0[0] + mv0[2] * mv0[2] + mv0[3] * mv0[3]
                        + mv0[4] * mv0[4] + mv0[8] * mv0[8] + mv0[9] * mv0[9]
                        + mv0[10] * mv0[10] + mv0[14] * mv0[14];
            float ip1 = mv1[0] * mv1[0] + mv1[2] * mv1[2] + mv1[3] * mv1[3]
                        + mv1[4] * mv1[4] + mv1[8] * mv1[8] + mv1[9] * mv1[9]
                        + mv1[10] * mv1[10] + mv1[14] * mv1[14];
            float ip2 = mv2[0] * mv2[0] + mv2[2] * mv2[2] + mv2[3] * mv2[3]
                        + mv2[4] * mv2[4] + mv2[8] * mv2[8] + mv2[9] * mv2[9]
                        + mv2[10] * mv2[10] + mv2[14] * mv2[14];
            float ip3 = mv3[0] * mv3[0] + mv3[2] * mv3[2] + mv3[3] * mv3[3]
                        + mv3[4] * mv3[4] + mv3[8] * mv3[8] + mv3[9] * mv3[9]
                        + mv3[10] * mv3[10] + mv3[14] * mv3[14];

            norm += ip0 + ip1 + ip2 + ip3;
        }

        for (size_t c = full_groups * 4; c < n_channels; ++c) {
            const float* mv = x_b + c * 16;
            float ip = mv[0] * mv[0] + mv[2] * mv[2] + mv[3] * mv[3]
                       + mv[4] * mv[4] + mv[8] * mv[8] + mv[9] * mv[9]
                       + mv[10] * mv[10] + mv[14] * mv[14];
            norm += ip;
        }

        norm *= inv_n_channels;

        float clamped = std::fmax(norm, eps);
        float scale   = 1.0f / std::sqrt(clamped);

        float* out_b = out_a + b * n_channels * 16;
        for (size_t c = 0; c < n_channels; ++c) {
            float w = (w_r != nullptr) ? w_r[c] : 1.0f;
            float cscale = scale * w;
            const float* x_c = x_b + c * 16;
            float* o_c = out_b + c * 16;
            for (int d = 0; d < 16; ++d)
                o_c[d] = x_c[d] * cscale;
        }
    }
}

}  // namespace turbogator
