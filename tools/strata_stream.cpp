#include "strata/engine/forward_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::filesystem::path model;
    std::filesystem::path labels;
    bool trace = false;
};

[[noreturn]] void usage(std::string_view problem = {}) {
    if (!problem.empty()) std::cerr << "strata_stream: " << problem << "\n\n";
    std::cerr << "Usage: strata_stream <model.onnx> --labels <labels.txt> [--trace]\n"
                 "Reads one comma-separated FP32 feature vector per stdin line and\n"
                 "writes one JSON inference result per stdout line.\n";
    std::exit(problem.empty() ? 0 : 2);
}

Options parse_options(int argc, char** argv) {
    if (argc < 2) usage("model is required");
    Options options{argv[1]};
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--trace") {
            options.trace = true;
        } else if (arg == "--labels" && ++i < argc) {
            options.labels = argv[i];
        } else {
            usage("unknown or incomplete option '" + std::string(arg) + "'");
        }
    }
    if (options.labels.empty()) usage("--labels is required");
    return options;
}

std::vector<float> parse_csv_row(const std::string& line) {
    std::vector<float> values;
    std::stringstream cells(line);
    std::string cell;
    while (std::getline(cells, cell, ',')) {
        const auto first = cell.find_first_not_of(" \t\r");
        if (first == std::string::npos) continue;
        const auto last = cell.find_last_not_of(" \t\r");
        std::size_t parsed = 0;
        const float value = std::stof(cell.substr(first, last - first + 1), &parsed);
        if (parsed != last - first + 1)
            throw std::runtime_error("invalid numeric value '" + cell + "'");
        values.push_back(value);
    }
    if (values.empty()) throw std::runtime_error("empty feature vector");
    return values;
}

std::vector<std::string> read_labels(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open labels file: " + path.string());
    std::vector<std::string> labels;
    std::string line;
    while (std::getline(input, line)) {
        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos) continue;
        const auto last = line.find_last_not_of(" \t\r");
        labels.push_back(line.substr(first, last - first + 1));
    }
    if (labels.empty()) throw std::runtime_error("labels file is empty");
    return labels;
}

std::string json_escape(std::string_view value) {
    std::string result;
    for (const char c : value) {
        if (c == '\\' || c == '"') {
            result.push_back('\\');
            result.push_back(c);
        } else if (c == '\n') result += "\\n";
        else if (c == '\r') result += "\\r";
        else if (c == '\t') result += "\\t";
        else result.push_back(c);
    }
    return result;
}

void write_error(std::string_view message) {
    std::cout << "{\"ok\":false,\"error\":\"" << json_escape(message) << "\"}\n" << std::flush;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto labels = read_labels(options.labels);
        Strata::Engine::ForwardEngine engine(options.model);
        if (labels.size() != engine.output_size()) {
            throw std::runtime_error("labels count " + std::to_string(labels.size()) +
                                     " does not match model outputs " +
                                     std::to_string(engine.output_size()));
        }

        std::vector<float> output(engine.output_size());
        const auto plan = options.trace ? engine.execution_plan()
                                        : std::vector<Strata::Engine::ExecutionStep>{};
        std::vector<double> trace_us(plan.size());
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                const auto input = parse_csv_row(line);
                if (input.size() != engine.input_size()) {
                    write_error("input has " + std::to_string(input.size()) +
                                " values; model expects " + std::to_string(engine.input_size()));
                    continue;
                }
                const auto started = std::chrono::steady_clock::now();
                if (options.trace) engine.forward_traced(input, output, trace_us);
                else engine.forward(input, output);
                const auto finished = std::chrono::steady_clock::now();
                const double elapsed_us =
                    std::chrono::duration<double, std::micro>(finished - started).count();

                const auto predicted = static_cast<std::size_t>(
                    std::distance(output.begin(), std::max_element(output.begin(), output.end())));
                const float max_logit = output[predicted];
                double denominator = 0.0;
                for (const float logit : output) denominator += std::exp(static_cast<double>(logit - max_logit));
                const double confidence = 1.0 / denominator;

                std::cout << std::setprecision(9)
                          << "{\"ok\":true,\"prediction_index\":" << predicted
                          << ",\"prediction\":\"" << json_escape(labels[predicted])
                          << "\",\"confidence\":" << confidence
                          << ",\"latency_us\":" << elapsed_us << ",\"scores\":[";
                for (std::size_t i = 0; i < output.size(); ++i)
                    std::cout << (i == 0 ? "" : ",") << output[i];
                std::cout << "]";
                if (options.trace) {
                    std::cout << ",\"step_trace_us\":[";
                    for (std::size_t i = 0; i < trace_us.size(); ++i)
                        std::cout << (i == 0 ? "" : ",") << trace_us[i];
                    std::cout << "]";
                }
                std::cout << "}\n" << std::flush;
            } catch (const std::exception& error) {
                write_error(error.what());
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "strata_stream: " << error.what() << "\n";
        return 1;
    }
}
