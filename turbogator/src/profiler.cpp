#include "profiler.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>
#include <x86intrin.h>

namespace {
class CycleTimer {
public:
    static inline uint64_t start() {
        _mm_lfence();
        return __rdtsc();
    }

    static inline uint64_t stop() {
        unsigned int aux;
        uint64_t cycles = __rdtscp(&aux);
        _mm_lfence();
        return cycles;
    }
};
}

double Profiler::benchmark_cycles(
    IGatrBackend* backend,
    const float* input,
    float* output,
    size_t batch,
    size_t N,
    size_t channels,
    int warmup_runs,
    int bench_runs
) {
    if (backend == nullptr || bench_runs <= 0) {
        return 0.0;
    }

    for (int i = 0; i < warmup_runs; ++i) {
        backend->forward(input, output, batch, N, channels);
    }

    std::vector<uint64_t> run_cycles;
    run_cycles.reserve(static_cast<size_t>(bench_runs));

    for (int i = 0; i < bench_runs; ++i) {
        uint64_t start_cycles = CycleTimer::start();
        backend->forward(input, output, batch, N, channels);
        uint64_t end_cycles = CycleTimer::stop();
        run_cycles.push_back(end_cycles - start_cycles);
    }

    std::sort(run_cycles.begin(), run_cycles.end());
    const size_t mid = run_cycles.size() / 2;

    if (run_cycles.size() % 2 == 0) {
        return (run_cycles[mid - 1] + run_cycles[mid]) / 2.0;
    }
    return static_cast<double>(run_cycles[mid]);
}