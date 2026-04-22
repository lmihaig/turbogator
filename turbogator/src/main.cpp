#include <cstddef>
#include <fstream>
#include <iostream>
#include <vector>
#include "backend_selection.hpp"
#include "profiler.hpp"

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <N> <batch> <channels>\n";
    return 1;
  }

  const size_t N = static_cast<size_t>(std::stoull(argv[1]));
  const size_t batch = static_cast<size_t>(std::stoull(argv[2]));
  const size_t channels = static_cast<size_t>(std::stoull(argv[3]));
  const size_t total_elements = batch * N * channels * 16;

  std::vector<float> input(total_elements, 1.0f);
  std::vector<float> output(total_elements, 0.0f);

  ActiveBackend backend = create_backend();

  std::ofstream out("metrics_temp.json", std::ios::trunc);
  if (!out.is_open()) {
    std::cerr << "Failed to open metrics_temp.json\n";
    return 1;
  }

  out << "{\"N\": " << N;
  const double avg_cycles = Profiler::benchmark_cycles(
      &backend, input.data(), output.data(), batch, N, channels);
  out << ", \"cycles\": " << avg_cycles;
  out << "}\n";

  return 0;
}
