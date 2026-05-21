// =============================================================================
//  Phase 5 — pipeline latency benchmark (C++ side)
// -----------------------------------------------------------------------------
//  Times the SAME model (models/mlp.onnx) on the SAME fixed input through:
//    * ONNX Runtime (C++)     — Strata::Onnx::OnnxModel       (Phase 3)
//    * StrataCompute (custom) — Strata::Engine::ForwardEngine (Phase 4)
//
//  Per-inference wall time is captured individually (not block-averaged) so
//  the P95/P99 tail — the metric that matters for HFT/edge — is real.
//  Results are written as CSV for the Python driver to merge with the
//  PyTorch baseline into docs/benchmarks.md. Both engines run single-thread
//  (ORT intra_op=1; the custom engine is inherently single-thread) for an
//  apples-to-apples latency comparison.
// =============================================================================
#include "strata/onnx/onnx_model.hpp"
#include "strata/engine/forward_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

namespace {

std::vector<float> csv_line(std::istream& is) {
    std::string line;
    std::getline(is, line);
    std::vector<float> v;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        if (!cell.empty()) v.push_back(std::stof(cell));
    }
    return v;
}

struct Stats {
    double mean, p50, p95, p99, min, max;
    std::size_t iters;
};

Stats summarise(std::vector<double>& us) {
    std::sort(us.begin(), us.end());
    const std::size_t n = us.size();
    auto q = [&](double p) {
        const std::size_t i = static_cast<std::size_t>(p * (n - 1) + 0.5);
        return us[i];
    };
    double sum = 0.0;
    for (double v : us) sum += v;
    return {sum / static_cast<double>(n), q(0.50), q(0.95), q(0.99),
            us.front(), us.back(), n};
}

template <class F>
Stats time_engine(F&& run_once, std::size_t warmup, std::size_t iters) {
    for (std::size_t i = 0; i < warmup; ++i) run_once();
    std::vector<double> us;
    us.reserve(iters);
    for (std::size_t i = 0; i < iters; ++i) {
        const auto t0 = clk::now();
        run_once();
        const auto t1 = clk::now();
        us.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    return summarise(us);
}

void print_row(const char* name, const Stats& s) {
    std::printf("%-22s %10zu %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
                name, s.iters, s.mean, s.p50, s.p95, s.p99, s.min, s.max);
}

} // namespace

int main(int argc, char** argv) {
    const fs::path models = (argc > 1) ? fs::path(argv[1]) : fs::path("models");
    const fs::path csv_out =
        (argc > 2) ? fs::path(argv[2]) : fs::path("results/latency_cpp.csv");
    const std::size_t warmup = (argc > 3) ? std::stoul(argv[3]) : 2000u;
    const std::size_t iters  = (argc > 4) ? std::stoul(argv[4]) : 50000u;

    std::ifstream io(models / "mlp_io.csv");
    if (!io) { std::printf("FATAL: cannot open mlp_io.csv\n"); return 2; }
    const std::vector<float> input    = csv_line(io);
    const std::vector<float> expected = csv_line(io);

    Strata::Onnx::OnnxModel       ort(models / "mlp.onnx", /*intra_op*/ 1);
    Strata::Engine::ForwardEngine eng(models / "mlp.onnx");

    std::vector<float> o_ort(ort.output_elements(0));
    std::vector<float> o_eng(eng.output_size());

    // Correctness gate before timing — never benchmark a wrong kernel.
    ort.run(input, o_ort);
    eng.forward(input, o_eng);
    double max_err = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double e_ort = std::fabs(
            static_cast<double>(o_ort[i]) - static_cast<double>(expected[i]));
        const double e_eng = std::fabs(
            static_cast<double>(o_eng[i]) - static_cast<double>(expected[i]));
        max_err = std::max({max_err, e_ort, e_eng});
    }
    std::printf("correctness vs numpy oracle: max_abs_err = %.3e  (%s)\n",
                max_err, max_err < 1e-3 ? "PASS" : "FAIL");
    if (max_err >= 1e-3) return 1;

    std::printf("\nwarmup=%zu  iters=%zu  (single-thread, microseconds)\n",
                warmup, iters);
    std::printf("%-22s %10s %10s %10s %10s %10s %10s %10s\n",
                "engine", "iters", "mean", "p50", "p95", "p99", "min", "max");

    const Stats s_ort = time_engine(
        [&] { ort.run(input, o_ort); }, warmup, iters);
    const Stats s_eng = time_engine(
        [&] { eng.forward(input, o_eng); }, warmup, iters);

    print_row("onnxruntime_cpp", s_ort);
    print_row("stratacompute_cpp", s_eng);

    // Analytical memory-traffic (cache-miss proxy) for the custom engine:
    // it touches exactly activations + I/O (+ weights, summed in the driver).
    const std::size_t eng_bytes =
        eng.activation_bytes() + eng.input_size() * sizeof(float) +
        eng.output_size() * sizeof(float);

    fs::create_directories(csv_out.parent_path());
    std::ofstream csv(csv_out);
    csv << "engine,iters,mean_us,p50_us,p95_us,p99_us,min_us,max_us,"
           "bytes_per_infer\n";
    auto emit = [&](const char* nm, const Stats& s, long long bytes) {
        csv << nm << ',' << s.iters << ',' << s.mean << ',' << s.p50 << ','
            << s.p95 << ',' << s.p99 << ',' << s.min << ',' << s.max << ','
            << bytes << '\n';
    };
    emit("onnxruntime_cpp", s_ort, -1);  // opaque runtime
    emit("stratacompute_cpp", s_eng, static_cast<long long>(eng_bytes));
    csv.close();

    std::printf("\nwrote %s\n", csv_out.string().c_str());
    return 0;
}
