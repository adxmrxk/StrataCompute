// Int8-weight / int8-activation execution for Strata's supported MLP graph.
// The model is quantized once at construction; forward() has no heap work.
#ifndef STRATA_ENGINE_QUANTIZED_FORWARD_ENGINE_HPP
#define STRATA_ENGINE_QUANTIZED_FORWARD_ENGINE_HPP

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>

namespace Strata::Engine {

class QuantizedForwardEngine {
public:
    explicit QuantizedForwardEngine(const std::filesystem::path& onnx_path);
    ~QuantizedForwardEngine();
    QuantizedForwardEngine(const QuantizedForwardEngine&) = delete;
    QuantizedForwardEngine& operator=(const QuantizedForwardEngine&) = delete;

    void forward(std::span<const float> input, std::span<float> output);
    [[nodiscard]] std::size_t input_size() const noexcept;
    [[nodiscard]] std::size_t output_size() const noexcept;
    /// Bytes in the deployed INT8 weights, FP32 biases, and layer scales.
    [[nodiscard]] std::size_t packed_model_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace Strata::Engine
#endif
