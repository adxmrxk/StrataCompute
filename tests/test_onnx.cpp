// Phase 3 — ONNX Runtime baseline integration test.
//
//  Standalone exe (separate from the offline Phase 1/2 suite because it needs
//  the downloaded ORT runtime + the generated model asset). argv[1] = the
//  models/ directory holding mlp.onnx and mlp_io.csv (numpy oracle).
#include "strata/onnx/onnx_model.hpp"
#include "test_harness.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Strata::Onnx::OnnxModel;

namespace {

fs::path g_models;  // set from argv before run_all()

std::vector<float> parse_csv_line(std::istream& is) {
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

STRATA_TEST(onnx_model_loads_and_introspects) {
    OnnxModel m(g_models / "mlp.onnx");
    STRATA_CHECK_EQ(m.num_inputs(), 1u);
    STRATA_CHECK_EQ(m.num_outputs(), 1u);
    STRATA_CHECK_EQ(m.input_name(0), std::string("input"));
    STRATA_CHECK_EQ(m.output_name(0), std::string("output"));
    STRATA_CHECK_EQ(m.input_elements(0), 16u);   // [1,16]
    STRATA_CHECK_EQ(m.output_elements(0), 8u);    // [1,8]
    // Static graph shape: [1, K] / [1, O].
    STRATA_CHECK_EQ(m.input_shape(0).size(), 2u);
    STRATA_CHECK_EQ(m.output_shape(0).back(), 8);
}

STRATA_TEST(onnx_run_matches_numpy_oracle) {
    std::ifstream io(g_models / "mlp_io.csv");
    STRATA_CHECK(io.good());
    const std::vector<float> input    = parse_csv_line(io);
    const std::vector<float> expected = parse_csv_line(io);
    STRATA_CHECK_EQ(input.size(), 16u);
    STRATA_CHECK_EQ(expected.size(), 8u);

    OnnxModel m(g_models / "mlp.onnx");
    std::vector<float> out(m.output_elements(0), 0.0f);
    m.run(input, out);

    for (std::size_t i = 0; i < expected.size(); ++i) {
        STRATA_CHECK(std::fabs(out[i] - expected[i]) < 1e-4f);
    }
}

STRATA_TEST(onnx_run_is_deterministic) {
    std::ifstream io(g_models / "mlp_io.csv");
    const std::vector<float> input = parse_csv_line(io);

    OnnxModel m(g_models / "mlp.onnx");
    std::vector<float> a(m.output_elements(0)), b(m.output_elements(0));
    m.run(input, a);
    m.run(input, b);
    for (std::size_t i = 0; i < a.size(); ++i) {
        STRATA_CHECK_EQ(a[i], b[i]);              // bit-identical re-runs
    }
}

int main(int argc, char** argv) {
    g_models = (argc > 1) ? fs::path(argv[1]) : fs::path("models");
    std::printf("=== StrataCompute Phase 3 ONNX baseline (%s) ===\n",
                g_models.string().c_str());
    try {
        return strata_test::run_all();
    } catch (const std::exception& e) {
        std::printf("FATAL: %s\n", e.what());
        return 2;
    }
}
