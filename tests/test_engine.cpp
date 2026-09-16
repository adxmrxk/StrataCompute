// Phase 4 — custom forward-pass engine.
//
//  Proves the whole stack composes: the engine extracts weights from
//  models/mlp.onnx (Strata::Onnx::OnnxGraph) into Strata::Tensor/MemoryPool,
//  compiles the Gemm/Relu graph, and runs it via the Phase 2 Strata::Compute
//  SIMD kernels. Output must match the numpy oracle in mlp_io.csv — the same
//  oracle ONNX Runtime was validated against in Phase 3, so a match here
//  means engine == ORT transitively. Re-running many times exercises the
//  reset()-recycled activation arena (allocation-free hot path).
#include "strata/engine/forward_engine.hpp"
#include "test_harness.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Strata::Engine::ForwardEngine;

#ifndef STRATA_MODELS_DIR
#  define STRATA_MODELS_DIR "models"
#endif

namespace {

const fs::path g_models{STRATA_MODELS_DIR};

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

} // namespace

STRATA_TEST(engine_loads_and_compiles_graph) {
    ForwardEngine eng(g_models / "mlp.onnx");
    STRATA_CHECK_EQ(eng.input_size(), 16u);
    STRATA_CHECK_EQ(eng.output_size(), 8u);
    STRATA_CHECK_EQ(eng.num_steps(), 3u);          // Gemm, Relu, Gemm
    STRATA_CHECK(eng.activation_bytes() > 0u);     // arena reserved
}

STRATA_TEST(engine_matches_numpy_oracle) {
    std::ifstream io(g_models / "mlp_io.csv");
    STRATA_CHECK(io.good());
    const std::vector<float> input    = csv_line(io);
    const std::vector<float> expected = csv_line(io);
    STRATA_CHECK_EQ(input.size(), 16u);
    STRATA_CHECK_EQ(expected.size(), 8u);

    ForwardEngine eng(g_models / "mlp.onnx");
    std::vector<float> out(eng.output_size(), 0.0f);
    eng.forward(input, out);

    for (std::size_t i = 0; i < expected.size(); ++i) {
        STRATA_CHECK(std::fabs(out[i] - expected[i]) < 1e-3f);
    }
}

STRATA_TEST(engine_hot_path_is_repeatable) {
    std::ifstream io(g_models / "mlp_io.csv");
    const std::vector<float> input = csv_line(io);

    ForwardEngine eng(g_models / "mlp.onnx");
    std::vector<float> first(eng.output_size());
    eng.forward(input, first);

    // 1000 inferences through the recycled arena -> bit-identical every time.
    std::vector<float> out(eng.output_size());
    for (int rep = 0; rep < 1000; ++rep) {
        eng.forward(input, out);
        for (std::size_t i = 0; i < out.size(); ++i)
            STRATA_CHECK_EQ(out[i], first[i]);
    }
}

STRATA_TEST(engine_rejects_wrong_input_size) {
    ForwardEngine eng(g_models / "mlp.onnx");
    std::vector<float> bad(15, 0.0f);              // expects 16
    std::vector<float> out(eng.output_size());
    STRATA_CHECK_THROWS(eng.forward(bad, out), std::runtime_error);
}

STRATA_TEST(engine_exposes_plan_and_diagnostic_trace) {
    std::ifstream io(g_models / "mlp_io.csv");
    const std::vector<float> input = csv_line(io);
    ForwardEngine eng(g_models / "mlp.onnx");
    const auto plan = eng.execution_plan();
    STRATA_CHECK_EQ(plan.size(), 3u);
    STRATA_CHECK(plan[0].operation == Strata::Engine::OperationKind::Gemm);
    STRATA_CHECK_EQ(plan[0].rows, 32u);
    STRATA_CHECK_EQ(plan[0].columns, 16u);
    STRATA_CHECK(plan[0].has_bias);
    STRATA_CHECK(plan[1].operation == Strata::Engine::OperationKind::Relu);
    STRATA_CHECK_EQ(plan[2].rows, 8u);

    std::vector<float> output(eng.output_size());
    std::vector<double> trace(plan.size());
    eng.forward_traced(input, output, trace);
    for (const double elapsed : trace) STRATA_CHECK(elapsed >= 0.0);
    STRATA_CHECK_THROWS(eng.forward_traced(input, output,
                                            std::span<double>(trace.data(), 2)),
                        std::runtime_error);
}
