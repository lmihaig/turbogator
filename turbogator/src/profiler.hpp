#pragma once

#include "gatr_interface.hpp"

#include <cstddef>

class Profiler {
public:
    static double benchmark_cycles(
        IGatrBackend* backend,
        const float* input,
        float* output,
        size_t batch,
        size_t N,
        size_t channels,
        int warmup_runs = 3,
        int bench_runs = 10
    );
};