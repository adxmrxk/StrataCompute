// =============================================================================
//  Strata::Engine::ForwardEngine  —  custom pure-C++ forward pass
// -----------------------------------------------------------------------------
//  Phase 4: ties Phases 1–3 together.
//
//   * Weights are extracted from the .onnx file (Strata::Onnx::OnnxGraph) at
//     construction and copied into Strata::Tensor structures backed by a
//     Strata::MemoryPool — no per-inference weight allocation.
//   * The graph topology (Gemm / Relu node sequence) is read from the same
//     file and compiled into a flat execution plan.
//   * forward() runs that plan in pure C++, invoking the Strata::Compute SIMD
//     operators, with all intermediate activations carved from a pre-allocated
//     activation arena that is reset()-recycled each call. The forward hot
//     path performs ZERO heap allocations.
//
//  Supported op set: Gemm (transB=1, optional bias) and Relu — i.e. the
//  feed-forward MLP family in the spec. Weight extraction itself is generic
//  over all float initializers; ONNX Runtime is not required (this builds and
//  runs in the default offline tree).
// =============================================================================
#ifndef STRATA_ENGINE_FORWARD_ENGINE_HPP
#define STRATA_ENGINE_FORWARD_ENGINE_HPP

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>

namespace Strata::Engine {

class ForwardEngine {
public:
    /// Parse the model, load weights into the pool, compile the plan.
    /// Throws std::runtime_error on unsupported ops / malformed models.
    explicit ForwardEngine(const std::filesystem::path& onnx_path);
    ~ForwardEngine();

    ForwardEngine(const ForwardEngine&)            = delete;
    ForwardEngine& operator=(const ForwardEngine&) = delete;

    /// Run inference. `input.size()` must equal input_size(); `output` must
    /// hold at least output_size() elements. Allocation-free, single-sample.
    void forward(std::span<const float> input, std::span<float> output);

    [[nodiscard]] std::size_t input_size()  const noexcept;
    [[nodiscard]] std::size_t output_size() const noexcept;
    /// Number of execution steps (Gemm/Relu) in the compiled plan.
    [[nodiscard]] std::size_t num_steps()   const noexcept;
    /// Bytes reserved for the activation arena (peak intermediate footprint).
    [[nodiscard]] std::size_t activation_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace Strata::Engine

#endif // STRATA_ENGINE_FORWARD_ENGINE_HPP
