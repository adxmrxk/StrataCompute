// =============================================================================
//  Strata::Onnx::OnnxGraph  —  minimal ONNX protobuf reader
// -----------------------------------------------------------------------------
//  Phase 4 needs the *raw weight tensors* and the *node topology* out of the
//  serialised .onnx file so the custom engine can run the graph itself. We do
//  NOT depend on ONNX Runtime or the protobuf/onnx C++ libraries for this:
//  ORT's inference API does not surface initializers, and pulling protobuf is
//  heavy. Instead this is a tight, dependency-free reader of just the subset
//  of the ONNX protobuf we need (ModelProto -> GraphProto -> node/initializer).
//
//  Scope: float tensors (data_type FLOAT) stored as `raw_data`, the form the
//  onnx Python exporter uses. Init-time only (not the hot path, not under
//  Strata::Compute) so std::vector/std::string are fine here.
// =============================================================================
#ifndef STRATA_ONNX_GRAPH_HPP
#define STRATA_ONNX_GRAPH_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Strata::Onnx {

/// A graph initializer (weight/bias constant) as stored in the file.
struct Initializer {
    std::string                name;
    std::vector<std::int64_t>  dims;
    std::span<const float>     data;   // view into the owned file buffer
};

/// One graph node (operator instance).
struct Node {
    std::string              op_type;     // "Gemm", "Relu", ...
    std::vector<std::string> input;       // tensor names consumed
    std::vector<std::string> output;      // tensor names produced
    std::int64_t             trans_b = 0; // Gemm transB attribute (0/1)
};

class OnnxGraph {
public:
    /// Parse the model file. Throws std::runtime_error on malformed input
    /// or unsupported encodings.
    explicit OnnxGraph(const std::filesystem::path& path);

    [[nodiscard]] const std::vector<Node>&        nodes()        const noexcept { return nodes_; }
    [[nodiscard]] const std::vector<Initializer>& initializers() const noexcept { return inits_; }
    [[nodiscard]] const std::string&              graph_input()  const noexcept { return input_; }
    [[nodiscard]] const std::string&              graph_output() const noexcept { return output_; }

    /// Initializer by name, or nullptr if absent.
    [[nodiscard]] const Initializer* find(std::string_view name) const noexcept;

private:
    std::vector<std::byte>     blob_;   // owns the raw file bytes
    std::vector<Node>          nodes_;
    std::vector<Initializer>   inits_;
    std::string                input_;
    std::string                output_;
};

} // namespace Strata::Onnx

#endif // STRATA_ONNX_GRAPH_HPP
