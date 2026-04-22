#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "backend_selection.hpp"

namespace {

void read_bin(const char* filename, float* data, size_t num_elements) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        exit(1);
    }
    file.read(reinterpret_cast<char*>(data), num_elements * sizeof(float));
}

std::string read_text_file(const char* filename) {
    std::ifstream file(filename);
    if (!file) {
        return "";
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

size_t read_validation_n(const char* filename) {
    const std::string content = read_text_file(filename);
    if (content.empty()) {
        std::cerr << "Error: missing or empty validation config: " << filename
                  << std::endl;
        exit(1);
    }

    const std::regex pattern("\"N\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (std::regex_search(content, match, pattern) && match.size() > 1) {
        try {
            return static_cast<size_t>(std::stoull(match[1].str()));
        } catch (...) {
            std::cerr << "Error: invalid N value in validation config: " << filename
                      << std::endl;
            exit(1);
        }
    }

    std::cerr << "Error: could not parse N from validation config: " << filename
              << std::endl;
    exit(1);
}

}

int main() {
    const size_t batch = 8;
    const size_t N = read_validation_n("../results/baseline/validation_config.json");
    const size_t channels = 2;
    const size_t total_elements = batch * N * channels * 16;

    std::vector<float> input(total_elements, 0.0f);
    std::vector<float> expected(total_elements, 0.0f);
    std::vector<float> actual(total_elements, 0.0f);

    std::cout << "Loading validation data for N=" << N << "..." << std::endl;
    read_bin("../results/baseline/input.bin", input.data(), total_elements);
    read_bin("../results/baseline/expected.bin", expected.data(), total_elements);

    ActiveBackend backend = create_backend();
    std::cout << "Validating backend..." << std::endl;
    backend.forward(input.data(), actual.data(), batch, N, channels);

    std::cout << "Comparing outputs..." << std::endl;
    double max_error = 0.0;
    for (size_t i = 0; i < total_elements; ++i) {
        const double err = std::abs(expected[i] - actual[i]);
        if (err > max_error) {
            max_error = err;
        }
    }

    std::cout << "Max Floating Point Error: " << max_error << std::endl;

    if (max_error < 1e-4) {
        std::cout << "VALIDATION PASSED" << std::endl;
    } else {
        std::cout << "VALIDATION FAILED. Check your math." << std::endl;
    }

    return max_error < 1e-4 ? 0 : 1;
}
