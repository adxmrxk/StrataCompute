#include "strata/engine/quantized_forward_engine.hpp"

#include "strata/compute/elementwise.hpp"
#include "strata/compute/matvec.hpp"
#include "strata/onnx/onnx_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Strata::Engine {

struct QuantizedForwardEngine::Impl {
    struct Ref { enum Kind { In, Out, Act } kind = Act; std::size_t slot = 0; };
    struct Step {
        enum Op { Gemm, Relu } op = Gemm;
        Ref in, out;
        std::vector<std::int8_t> weights;
        std::vector<float> bias;
        std::size_t M = 0, K = 0, n = 0;
        float weight_scale = 1.0F;
    };

    Onnx::OnnxGraph graph;
    std::vector<Step> plan;
    std::vector<std::vector<float>> activations;
    std::vector<std::size_t> act_sizes;
    std::vector<float*> act_ptr;
    std::vector<std::int8_t> quantized_input;
    std::size_t in_size = 0, out_size = 0, packed_bytes = 0;

    explicit Impl(const std::filesystem::path& path) : graph(path) { build(); }

    static std::pair<std::vector<std::int8_t>, float> quantize_weights(std::span<const float> source) {
        float maximum = 0.0F;
        for (float value : source) maximum = std::max(maximum, std::abs(value));
        const float scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
        std::vector<std::int8_t> result(source.size());
        for (std::size_t i = 0; i < source.size(); ++i) {
            const int value = static_cast<int>(std::lround(source[i] / scale));
            result[i] = static_cast<std::int8_t>(std::clamp(value, -127, 127));
        }
        return {std::move(result), scale};
    }

    void build() {
        const auto& nodes = graph.nodes();
        const auto& gin = graph.graph_input();
        const auto& gout = graph.graph_output();
        std::unordered_map<std::string, std::size_t> elems;
        for (const auto& initializer : graph.initializers()) elems[initializer.name] = initializer.data.size();
        bool found_input = false;
        for (const auto& node : nodes) {
            if (node.op_type == "Gemm" && node.input.size() >= 2 && node.input[0] == gin) {
                const auto* weight = graph.find(node.input[1]);
                if (!weight || weight->dims.size() != 2) throw std::runtime_error("QuantizedForwardEngine: bad first Gemm");
                elems[gin] = static_cast<std::size_t>(weight->dims[1]);
                found_input = true;
                break;
            }
        }
        if (!found_input) throw std::runtime_error("QuantizedForwardEngine: cannot infer input size");
        std::unordered_map<std::string, std::size_t> slotmap;
        auto ref = [&](const std::string& name) -> Ref {
            if (name == gin) return {Ref::In, 0};
            if (name == gout) return {Ref::Out, 0};
            if (graph.find(name)) throw std::runtime_error("QuantizedForwardEngine: initializer data input unsupported");
            const auto [it, inserted] = slotmap.emplace(name, act_sizes.size());
            if (inserted) act_sizes.push_back(elems.at(name));
            return {Ref::Act, it->second};
        };
        for (const auto& node : nodes) {
            if (node.op_type == "Gemm") {
                if (node.input.size() < 2 || node.output.empty() || node.trans_b != 1)
                    throw std::runtime_error("QuantizedForwardEngine: unsupported Gemm");
                const auto* weight = graph.find(node.input[1]);
                if (!weight || weight->dims.size() != 2) throw std::runtime_error("QuantizedForwardEngine: bad Gemm weight");
                Step step;
                step.M = static_cast<std::size_t>(weight->dims[0]);
                step.K = static_cast<std::size_t>(weight->dims[1]);
                if (elems.at(node.input[0]) != step.K) throw std::runtime_error("QuantizedForwardEngine: Gemm shape mismatch");
                auto quantized = quantize_weights(weight->data);
                step.weights = std::move(quantized.first);
                step.weight_scale = quantized.second;
                if (node.input.size() >= 3 && !node.input[2].empty()) {
                    const auto* bias = graph.find(node.input[2]);
                    if (!bias || bias->data.size() != step.M) throw std::runtime_error("QuantizedForwardEngine: bad Gemm bias");
                    step.bias.assign(bias->data.begin(), bias->data.end());
                }
                elems[node.output[0]] = step.M;
                step.in = ref(node.input[0]);
                step.out = ref(node.output[0]);
                packed_bytes += step.weights.size() + step.bias.size() * sizeof(float) + sizeof(float);
                quantized_input.resize(std::max(quantized_input.size(), step.K));
                plan.push_back(std::move(step));
            } else if (node.op_type == "Relu") {
                if (node.input.empty() || node.output.empty()) throw std::runtime_error("QuantizedForwardEngine: malformed Relu");
                Step step; step.op = Step::Relu; step.n = elems.at(node.input[0]);
                elems[node.output[0]] = step.n; step.in = ref(node.input[0]); step.out = ref(node.output[0]);
                plan.push_back(std::move(step));
            } else {
                throw std::runtime_error("QuantizedForwardEngine: unsupported op '" + node.op_type + "'");
            }
        }
        in_size = elems.at(gin); out_size = elems.at(gout);
        activations.reserve(act_sizes.size()); act_ptr.resize(act_sizes.size());
        for (std::size_t size : act_sizes) activations.emplace_back(size);
    }

    float* resolve(const Ref& ref, std::span<const float> input, std::span<float> output) noexcept {
        if (ref.kind == Ref::In) return const_cast<float*>(input.data());
        if (ref.kind == Ref::Out) return output.data();
        return act_ptr[ref.slot];
    }

    void forward(std::span<const float> input, std::span<float> output) {
        if (input.size() != in_size || output.size() < out_size)
            throw std::runtime_error("QuantizedForwardEngine::forward: tensor size");
        for (std::size_t i = 0; i < activations.size(); ++i) act_ptr[i] = activations[i].data();
        for (const Step& step : plan) {
            const float* source = resolve(step.in, input, output);
            float* destination = resolve(step.out, input, output);
            if (step.op == Step::Relu) {
                Compute::relu(std::span<float>(destination, step.n), source, step.n);
                continue;
            }
            float maximum = 0.0F;
            for (std::size_t i = 0; i < step.K; ++i) maximum = std::max(maximum, std::abs(source[i]));
            const float input_scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
            for (std::size_t i = 0; i < step.K; ++i) {
                const int value = static_cast<int>(std::lround(source[i] / input_scale));
                quantized_input[i] = static_cast<std::int8_t>(std::clamp(value, -127, 127));
            }
            Compute::matvec_i8(std::span<float>(destination, step.M), step.weights.data(),
                               quantized_input.data(), step.M, step.K, input_scale * step.weight_scale);
            if (!step.bias.empty()) Compute::add_inplace(std::span<float>(destination, step.M), step.bias.data(), step.M);
        }
    }
};

QuantizedForwardEngine::QuantizedForwardEngine(const std::filesystem::path& path) : p_(std::make_unique<Impl>(path)) {}
QuantizedForwardEngine::~QuantizedForwardEngine() = default;
void QuantizedForwardEngine::forward(std::span<const float> input, std::span<float> output) { p_->forward(input, output); }
std::size_t QuantizedForwardEngine::input_size() const noexcept { return p_->in_size; }
std::size_t QuantizedForwardEngine::output_size() const noexcept { return p_->out_size; }
std::size_t QuantizedForwardEngine::packed_model_bytes() const noexcept { return p_->packed_bytes; }

} // namespace Strata::Engine
