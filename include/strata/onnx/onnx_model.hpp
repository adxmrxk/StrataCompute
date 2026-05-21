// =============================================================================
//  Strata::Onnx::OnnxModel  —  ONNX Runtime C++ API wrapper
// -----------------------------------------------------------------------------
//  Phase 3 deliverable. Provides:
//    * a model loader that ingests a serialised .onnx file,
//    * graph I/O introspection (names + shapes),
//    * an execution runner mapping a flat input to a flat output.
//
//  Scope note: this is the spec's *dual-execution benchmark baseline*, not the
//  StrataCompute hot path. It deliberately lives under `Strata::Onnx` (NOT
//  `Strata::Compute`) — ONNX Runtime manages its own memory internally, which
//  is fine here precisely because the no-allocation contract is scoped to
//  `Strata::Compute`. In Phase 5 this baseline is timed head-to-head against
//  the custom engine.
// =============================================================================
#ifndef STRATA_ONNX_MODEL_HPP
#define STRATA_ONNX_MODEL_HPP

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace Strata::Onnx {

class OnnxModel {
public:
    /// Load and JIT-prepare the model. `intra_op_threads` pins the ORT thread
    /// pool (1 = deterministic single-thread, fairest baseline for latency).
    /// Throws Ort::Exception / std::runtime_error on failure.
    explicit OnnxModel(const std::filesystem::path& model_path,
                       int intra_op_threads = 1);
    ~OnnxModel();

    OnnxModel(const OnnxModel&)            = delete;
    OnnxModel& operator=(const OnnxModel&) = delete;

    [[nodiscard]] std::size_t num_inputs()  const noexcept;
    [[nodiscard]] std::size_t num_outputs() const noexcept;

    [[nodiscard]] const std::string& input_name(std::size_t i)  const;
    [[nodiscard]] const std::string& output_name(std::size_t i) const;

    /// Static graph shape for input/output `i`; dynamic dims are reported -1.
    [[nodiscard]] const std::vector<std::int64_t>& input_shape(std::size_t i) const;
    [[nodiscard]] const std::vector<std::int64_t>& output_shape(std::size_t i) const;

    /// Element count of input/output `i` with any dynamic batch dim treated
    /// as 1 (the single-sample inference case used by the baseline).
    [[nodiscard]] std::size_t input_elements(std::size_t i = 0)  const;
    [[nodiscard]] std::size_t output_elements(std::size_t i = 0) const;

    /// Run the single-input / single-output float graph: feeds `input`
    /// (row-major, length == input_elements()) and writes `output`
    /// (length == output_elements()). Throws on shape/size mismatch.
    void run(std::span<const float> input, std::span<float> output);

private:
    struct Impl;                 // pImpl hides all <onnxruntime_cxx_api.h>
    std::unique_ptr<Impl> p_;
};

} // namespace Strata::Onnx

#endif // STRATA_ONNX_MODEL_HPP
