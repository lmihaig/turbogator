#pragma once

#include "gatr_interface.hpp"

class FakeBackend final : public IGatrBackend {
public:
    void forward(
        const float* input,
        float* output,
        size_t batch,
        size_t N,
        size_t channels
    ) override {
        const size_t total_elements = batch * N * channels * 16;

        __asm__ volatile("# LLVM-MCA-BEGIN");

        for (size_t i = 0; i < total_elements; ++i) {
            output[i] = input[i] * 1.0f;
        }
        __asm__ volatile("# LLVM-MCA-END");
    }
};