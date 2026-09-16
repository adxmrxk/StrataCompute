#include "strata/engine/forward_engine.hpp"
#include "strata/engine/quantized_forward_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

std::vector<std::vector<float>> read_rows(const std::filesystem::path& file) {
    std::ifstream stream(file);
    if (!stream) throw std::runtime_error("cannot open input: " + file.string());
    std::vector<std::vector<float>> rows; std::string line;
    while (std::getline(stream, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream values(line); std::vector<float> row; float value = 0.0F;
        while (values >> value) row.push_back(value);
        if (!row.empty()) rows.push_back(std::move(row));
    }
    if (rows.empty()) throw std::runtime_error("input has no numbers");
    return rows;
}

std::vector<int> read_labels(const std::filesystem::path& file) {
    std::ifstream stream(file); if (!stream) throw std::runtime_error("cannot open labels: " + file.string());
    std::vector<int> labels; int label = 0; while (stream >> label) labels.push_back(label - 1);
    if (labels.empty()) throw std::runtime_error("labels are empty"); return labels;
}

double percentile(std::vector<double> samples, double p) {
    std::sort(samples.begin(), samples.end());
    return samples[static_cast<std::size_t>(std::ceil(p * samples.size())) - 1];
}

void usage() {
    std::cerr << "Usage: strata_int8_infer <model.onnx> <input.csv> [--iterations N] [--verify-fp32] [--labels y.txt]\n"
              << "Runs dynamic-activation INT8 execution with packed INT8 weights.\n";
}
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) { usage(); return 2; }
        std::size_t iterations = 1; bool verify = false; std::filesystem::path labels;
        for (int i = 3; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--verify-fp32") verify = true;
            else if (arg == "--iterations" && i + 1 < argc) iterations = std::stoull(argv[++i]);
            else if (arg == "--labels" && i + 1 < argc) labels = argv[++i];
            else { usage(); return 2; }
        }
        const auto inputs = read_rows(argv[2]); const auto& input = inputs.front();
        Strata::Engine::QuantizedForwardEngine engine(argv[1]);
        std::vector<float> output(engine.output_size());
        if (input.size() != engine.input_size()) throw std::runtime_error("input width does not match model");
        engine.forward(input, output); // warm up dispatch/cache
        std::vector<double> times; times.reserve(iterations);
        for (std::size_t i = 0; i < iterations; ++i) {
            const auto start = Clock::now(); engine.forward(input, output); const auto finish = Clock::now();
            times.push_back(std::chrono::duration<double, std::micro>(finish - start).count());
        }
        double mean = 0.0; for (double sample : times) mean += sample; mean /= times.size();
        std::cout << std::fixed << std::setprecision(4)
                  << "INT8 packed model: " << engine.packed_model_bytes() / 1024.0 << " KiB\n"
                  << "INT8 inference: mean " << mean << " us, p99 " << percentile(times, .99) << " us\n";
        if (verify) {
            Strata::Engine::ForwardEngine fp32(argv[1]);
            std::vector<float> reference(fp32.output_size()); fp32.forward(input, reference);
            float error = 0.0F;
            for (std::size_t i = 0; i < output.size(); ++i) error = std::max(error, std::abs(output[i] - reference[i]));
            std::cout << "FP32 comparison max_abs_delta: " << error << "\n";
        }
        if (!labels.empty()) {
            const auto expected = read_labels(labels);
            if (expected.size() != inputs.size()) throw std::runtime_error("label count does not match input rows");
            std::size_t correct = 0;
            for (std::size_t row = 0; row < inputs.size(); ++row) {
                if (inputs[row].size() != engine.input_size()) throw std::runtime_error("batch input width does not match model");
                engine.forward(inputs[row], output);
                const auto predicted = static_cast<int>(std::max_element(output.begin(), output.end()) - output.begin());
                correct += predicted == expected[row] ? 1U : 0U;
            }
            std::cout << "INT8 held-out accuracy: " << (100.0 * static_cast<double>(correct) / inputs.size())
                      << "% (" << correct << "/" << inputs.size() << ")\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "strata_int8_infer: " << error.what() << '\n'; return 1;
    }
}
