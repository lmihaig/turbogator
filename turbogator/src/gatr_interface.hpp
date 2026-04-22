#pragma once

#include <cstddef>

class IGatrBackend {
public:
    virtual ~IGatrBackend() = default;
    virtual void forward(
        const float* input,
        float* output,
        size_t batch,
        size_t N,
        size_t channels
    ) = 0;
};