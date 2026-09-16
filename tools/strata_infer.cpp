#include "strata/compute/cpu_features.hpp"
#include "strata/compute/matvec.hpp"
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

using Clock = std::chrono::steady_clock;

struct Options {
    std::filesystem::path model;
    std::filesystem::path input_csv;
    Strata::Compute::Backend backend = Strata::Compute::Backend::Auto;
    std::size_t iterations = 1;
    std::size_t verify_repeats = 0;
    bool oracle = false;
    bool json = false;
    bool profile = false;
    bool compare_backends = false;
    bool trace = false;
};

struct Timing {
    double mean_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    double p99_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
};

[[noreturn]] void usage(std::string_view error = {}) {
    if (!error.empty()) std::cerr << "error: " << error << "\n\n";
    std::cerr
        << "Usage: strata_infer <model.onnx> <input.csv> [options]\n\n"
        << "Runs StrataCompute's custom, ONNX-Runtime-free forward engine.\n"
        << "The CSV input's first row is the input vector; --oracle compares\n"
        << "the second row against the model output.\n\n"
        << "Options:\n"
        << "  --backend auto|scalar|avx2|avx512  Select a kernel policy (default: auto)\n"
        << "  --iterations N                     Measure N inferences (default: 1)\n"
        << "  --verify-repeats N                 Require N bit-identical repeat runs\n"
        << "  --oracle                           Validate output against CSV row two\n"
        << "  --profile                          Report p50/p95/p99 latency\n"
        << "  --compare-backends                 Compare scalar and auto kernels\n"
        << "  --trace                            Show compiled plan and per-step timing\n"
        << "  --json                             Emit one machine-readable result\n";
    std::exit(error.empty() ? 0 : 2);
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
            throw std::runtime_error("invalid CSV value '" + cell + "'");
        values.push_back(value);
    }
    if (values.empty()) throw std::runtime_error("CSV row contains no values");
    return values;
}

std::vector<std::vector<float>> read_rows(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open input CSV: " + path.string());
    std::vector<std::vector<float>> rows;
    std::string line;
    while (std::getline(input, line)) {
        if (line.find_first_not_of(" \t\r") != std::string::npos)
            rows.push_back(parse_csv_row(line));
    }
    if (rows.empty()) throw std::runtime_error("input CSV is empty");
    return rows;
}

Strata::Compute::Backend parse_backend(std::string_view value) {
    using Strata::Compute::Backend;
    if (value == "auto") return Backend::Auto;
    if (value == "scalar") return Backend::Scalar;
    if (value == "avx2") return Backend::AVX2;
    if (value == "avx512") return Backend::AVX512;
    usage("unknown backend '" + std::string(value) + "'");
}

std::string_view backend_name(Strata::Compute::Backend backend) {
    using Strata::Compute::Backend;
    switch (backend) {
        case Backend::Scalar: return "scalar";
        case Backend::AVX2: return "avx2";
        case Backend::AVX512: return "avx512";
        case Backend::Auto: return "auto";
    }
    return "unknown";
}

std::string json_escape(std::string_view value) {
    std::string result;
    for (const char c : value) {
        if (c == '\\' || c == '\"') {
            result.push_back('\\');
            result.push_back(c);
        } else if (c == '\n') {
            result += "\\n";
        } else if (c == '\r') {
            result += "\\r";
        } else if (c == '\t') {
            result += "\\t";
        } else {
            result.push_back(c);
        }
    }
    return result;
}

std::size_t parse_positive(std::string_view text, std::string_view name) {
    std::size_t parsed = 0;
    const auto value = std::stoull(std::string(text), &parsed);
    if (parsed != text.size() || value == 0 ||
        value > std::numeric_limits<std::size_t>::max())
        usage(std::string(name) + " must be a positive integer");
    return static_cast<std::size_t>(value);
}

Options parse_options(int argc, char** argv) {
    if (argc < 3) usage("model and input CSV are required");
    Options options{argv[1], argv[2]};
    for (int i = 3; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--json") options.json = true;
        else if (arg == "--oracle") options.oracle = true;
        else if (arg == "--profile") options.profile = true;
        else if (arg == "--compare-backends") options.compare_backends = true;
        else if (arg == "--trace") options.trace = true;
        else if (arg == "--backend" && ++i < argc) options.backend = parse_backend(argv[i]);
        else if (arg == "--iterations" && ++i < argc)
            options.iterations = parse_positive(argv[i], "--iterations");
        else if (arg == "--verify-repeats" && ++i < argc)
            options.verify_repeats = parse_positive(argv[i], "--verify-repeats");
        else usage("unknown or incomplete option '" + std::string(arg) + "'");
    }
        return options;
}

std::string_view operation_name(Strata::Engine::OperationKind operation) {
    return operation == Strata::Engine::OperationKind::Gemm ? "gemm" : "relu";
}

Timing measure(Strata::Engine::ForwardEngine& engine,
               const std::vector<float>& input,
               std::vector<float>& output,
               std::size_t iterations) {
    std::vector<double> samples;
    samples.reserve(iterations);
    for (std::size_t i = 0; i < iterations; ++i) {
        const auto start = Clock::now();
        engine.forward(input, output);
        const auto finish = Clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(finish - start).count());
    }
    std::sort(samples.begin(), samples.end());
    double sum = 0.0;
    for (const double sample : samples) sum += sample;
    const auto percentile = [&](double fraction) {
        const auto index = static_cast<std::size_t>(fraction * (samples.size() - 1) + 0.5);
        return samples[index];
    };
    return {sum / static_cast<double>(samples.size()), percentile(0.50),
            percentile(0.95), percentile(0.99), samples.front(), samples.back()};
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto rows = read_rows(options.input_csv);
        if (options.oracle && rows.size() < 2)
            usage("--oracle requires a second CSV row");

        Strata::Compute::set_backend(options.backend);
        Strata::Engine::ForwardEngine engine(options.model);
        if (rows.front().size() != engine.input_size())
            throw std::runtime_error("input size is " + std::to_string(rows.front().size()) +
                                     ", model expects " + std::to_string(engine.input_size()));

        std::vector<float> output(engine.output_size());
        engine.forward(rows.front(), output); // warmup and user-visible result

        const std::vector<Strata::Engine::ExecutionStep> plan =
            options.trace ? engine.execution_plan()
                          : std::vector<Strata::Engine::ExecutionStep>{};
        std::vector<double> trace_us(plan.size());
        if (options.trace) engine.forward_traced(rows.front(), output, trace_us);

        double max_error = 0.0;
        if (options.oracle) {
            if (rows[1].size() != output.size())
                throw std::runtime_error("oracle size does not match model output");
            for (std::size_t i = 0; i < output.size(); ++i)
                max_error = std::max(max_error, std::fabs(static_cast<double>(output[i] - rows[1][i])));
        }

        bool repeatable = true;
        std::vector<float> repeated(output.size());
        for (std::size_t i = 0; i < options.verify_repeats; ++i) {
            engine.forward(rows.front(), repeated);
            repeatable = repeatable && repeated == output;
        }

        const Timing selected_timing = measure(engine, rows.front(), repeated, options.iterations);

        double max_backend_delta = 0.0;
        Timing scalar_timing{};
        Timing auto_timing{};
        bool backend_equivalent = true;
        if (options.compare_backends) {
            std::vector<float> scalar_output(output.size());
            std::vector<float> auto_output(output.size());
            Strata::Compute::set_backend(Strata::Compute::Backend::Scalar);
            scalar_timing = measure(engine, rows.front(), scalar_output, options.iterations);
            Strata::Compute::set_backend(Strata::Compute::Backend::Auto);
            auto_timing = measure(engine, rows.front(), auto_output, options.iterations);
            for (std::size_t i = 0; i < output.size(); ++i)
                max_backend_delta = std::max(max_backend_delta,
                    std::fabs(static_cast<double>(scalar_output[i] - auto_output[i])));
            backend_equivalent = max_backend_delta < 1e-5;
            Strata::Compute::set_backend(options.backend);
        }

        const auto& cpu = Strata::Compute::cpu_features();
        const auto active = Strata::Compute::active_backend();
        const bool oracle_ok = !options.oracle || max_error < 1e-3;
        if (options.json) {
            std::cout << std::setprecision(9)
                      << "{\"model\":\"" << json_escape(options.model.string())
                      << "\",\"input_size\":" << engine.input_size()
                      << ",\"output_size\":" << engine.output_size()
                      << ",\"steps\":" << engine.num_steps()
                      << ",\"activation_bytes\":" << engine.activation_bytes()
                      << ",\"backend\":\"" << backend_name(active)
                      << "\",\"avx2_available\":" << (cpu.avx2 && cpu.fma ? "true" : "false")
                      << ",\"iterations\":" << options.iterations
                      << ",\"mean_us\":" << selected_timing.mean_us
                      << ",\"p50_us\":" << selected_timing.p50_us
                      << ",\"p95_us\":" << selected_timing.p95_us
                      << ",\"p99_us\":" << selected_timing.p99_us
                      << ",\"max_us\":" << selected_timing.max_us
                      << ",\"repeatable\":" << (repeatable ? "true" : "false")
                      << ",\"max_abs_error\":" << max_error
                      << ",\"backend_equivalent\":" << (backend_equivalent ? "true" : "false")
                      << ",\"backend_delta\":" << max_backend_delta
                      << ",\"trace\":[";
            for (std::size_t i = 0; i < plan.size(); ++i) {
                const auto& step = plan[i];
                std::cout << (i ? "," : "")
                          << "{\"operation\":\"" << operation_name(step.operation)
                          << "\",\"input_elements\":" << step.input_elements
                          << ",\"output_elements\":" << step.output_elements
                          << ",\"rows\":" << step.rows
                          << ",\"columns\":" << step.columns
                          << ",\"has_bias\":" << (step.has_bias ? "true" : "false")
                          << ",\"elapsed_us\":" << trace_us[i] << "}";
            }
            std::cout << "],\"output\":[";
            for (std::size_t i = 0; i < output.size(); ++i)
                std::cout << (i ? "," : "") << output[i];
            std::cout << "]}\n";
        } else {
            std::cout << std::fixed << std::setprecision(6)
                      << "StrataCompute inference\n"
                      << "  model: " << options.model.string() << "\n"
                      << "  graph: " << engine.input_size() << " inputs -> "
                      << engine.num_steps() << " steps -> " << engine.output_size() << " outputs\n"
                      << "  backend: " << backend_name(active)
                      << " (AVX2/FMA " << (cpu.avx2 && cpu.fma ? "available" : "unavailable") << ")\n"
                      << "  activation arena: " << engine.activation_bytes() << " bytes\n"
                      << "  timing: " << options.iterations << " iterations, mean "
                      << selected_timing.mean_us << " us, max " << selected_timing.max_us << " us\n";
            if (options.profile)
                std::cout << "  latency profile: p50 " << selected_timing.p50_us
                          << " us, p95 " << selected_timing.p95_us
                          << " us, p99 " << selected_timing.p99_us << " us\n";
            if (options.verify_repeats)
                std::cout << "  determinism: " << (repeatable ? "PASS" : "FAIL")
                          << " (" << options.verify_repeats << " repeat runs)\n";
            if (options.oracle)
                std::cout << "  oracle: " << (oracle_ok ? "PASS" : "FAIL")
                          << " (max abs error " << max_error << ")\n";
            if (options.compare_backends)
                std::cout << "  backend comparison: "
                          << (backend_equivalent ? "PASS" : "FAIL")
                          << " (scalar " << scalar_timing.mean_us << " us, auto "
                          << auto_timing.mean_us << " us, max delta "
                          << max_backend_delta << ")\n";
            if (options.trace) {
                std::cout << "  execution trace:\n";
                for (std::size_t i = 0; i < plan.size(); ++i) {
                    const auto& step = plan[i];
                    std::cout << "    " << i << ": " << operation_name(step.operation)
                              << " " << step.input_elements << " -> "
                              << step.output_elements;
                    if (step.operation == Strata::Engine::OperationKind::Gemm)
                        std::cout << " (W=" << step.rows << "x" << step.columns
                                  << ", bias=" << (step.has_bias ? "yes" : "no") << ")";
                    std::cout << ", diagnostic timing " << trace_us[i] << " us\n";
                }
            }
            std::cout << "  output: [";
            for (std::size_t i = 0; i < output.size(); ++i)
                std::cout << (i ? ", " : "") << output[i];
            std::cout << "]\n";
        }
        return oracle_ok && repeatable && backend_equivalent ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "strata_infer: " << error.what() << "\n";
        return 1;
    }
}
