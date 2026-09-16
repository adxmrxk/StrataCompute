// =============================================================================
//  Strata::Engine::ForwardEngine — implementation
// -----------------------------------------------------------------------------
//  Build (init-time): parse ONNX -> copy float initializers into Tensors
//  backed by a weights MemoryPool -> compile node list into a flat plan ->
//  size the activation arena to the peak intermediate footprint.
//
//  forward() (hot path): reset the activation arena, bump-carve one buffer
//  per intermediate, then execute the plan through Strata::Compute kernels.
//  No heap allocation occurs on this path.
// =============================================================================
#include "strata/engine/forward_engine.hpp"

#include "strata/onnx/onnx_graph.hpp"
#include "strata/memory_pool.hpp"
#include "strata/tensor.hpp"
#include "strata/compute/matvec.hpp"
#include "strata/compute/elementwise.hpp"

#include <cstring>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace Strata::Engine {
namespace {

constexpr std::size_t kAlign = Strata::kCacheLineBytes;

[[nodiscard]] std::size_t align_up(std::size_t n) noexcept {
    return (n + (kAlign - 1)) & ~(kAlign - 1);
}

} // namespace

struct ForwardEngine::Impl {
    // A plan operand: graph input, graph output, or an activation slot.
    struct Ref {
        enum Kind { In, Out, Act } kind = Act;
        std::size_t slot = 0;
    };
    struct Step {
        enum Op { Gemm, Relu } op = Gemm;
        Ref          in;
        Ref          out;
        const float* W = nullptr;   // Gemm weight  [M x K] row-major
        const float* B = nullptr;   // Gemm bias    [M] or null
        std::size_t  M = 0, K = 0;  // Gemm dims
        std::size_t  n = 0;         // Relu length
    };

    Onnx::OnnxGraph             graph;
    std::optional<MemoryPool>   weights_pool;
    std::optional<MemoryPool>   act_pool;
    std::vector<Tensor<float>>  weights;          // keep carved tensors alive
    std::vector<Step>           plan;
    std::vector<std::size_t>    act_elems;        // element count per slot
    std::vector<float*>         act_ptr;          // resolved each forward()
    std::size_t in_size = 0, out_size = 0, act_bytes = 0;

    explicit Impl(const std::filesystem::path& path) : graph(path) {
        build();
    }

    void build() {
        const auto& inits = graph.initializers();
        const auto& nodes = graph.nodes();

        // ---- weights: one Tensor per float initializer, copied into a pool
        std::size_t wcap = 0;
        for (const auto& t : inits) wcap += align_up(t.data.size() * sizeof(float));
        weights_pool.emplace(wcap ? wcap : kAlign);
        weights.reserve(inits.size());

        std::unordered_map<std::string, const float*> wmap;
        std::unordered_map<std::string, std::size_t>  elems;
        for (const auto& t : inits) {
            const std::size_t count = t.data.size();
            Tensor<float> ten = Tensor<float>::create(*weights_pool, {count});
            std::memcpy(ten.data(), t.data.data(), count * sizeof(float));
            weights.push_back(ten);
            wmap[t.name]  = ten.data();
            elems[t.name] = count;
        }

        // ---- infer graph-input element count from its first Gemm consumer
        const std::string& gin  = graph.graph_input();
        const std::string& gout = graph.graph_output();
        bool in_found = false;
        for (const auto& nd : nodes) {
            if (nd.op_type == "Gemm" && nd.input.size() >= 2 &&
                nd.input[0] == gin) {
                const Onnx::Initializer* w = graph.find(nd.input[1]);
                if (!w || w->dims.size() != 2)
                    throw std::runtime_error("ForwardEngine: bad Gemm weight");
                elems[gin] = static_cast<std::size_t>(w->dims[1]);  // transB=1
                in_found = true;
                break;
            }
        }
        if (!in_found)
            throw std::runtime_error("ForwardEngine: cannot infer input size");

        std::unordered_map<std::string, std::size_t> slotmap;
        auto ref = [&](const std::string& name) -> Ref {
            if (name == gin)  return {Ref::In,  0};
            if (name == gout) return {Ref::Out, 0};
            if (wmap.count(name))
                throw std::runtime_error(
                    "ForwardEngine: weight used as a data input (unsupported)");
            auto it = slotmap.find(name);
            std::size_t s;
            if (it == slotmap.end()) {
                s = act_elems.size();
                slotmap.emplace(name, s);
                act_elems.push_back(elems.at(name));
            } else {
                s = it->second;
            }
            return {Ref::Act, s};
        };

        // ---- compile nodes into the flat plan
        plan.reserve(nodes.size());
        for (const auto& nd : nodes) {
            if (nd.op_type == "Gemm") {
                if (nd.input.size() < 2 || nd.output.empty())
                    throw std::runtime_error("ForwardEngine: malformed Gemm");
                if (nd.trans_b != 1)
                    throw std::runtime_error(
                        "ForwardEngine: only Gemm transB=1 is supported");
                const Onnx::Initializer* w = graph.find(nd.input[1]);
                if (!w || w->dims.size() != 2)
                    throw std::runtime_error("ForwardEngine: bad Gemm weight");

                Step st;
                st.op = Step::Gemm;
                st.M  = static_cast<std::size_t>(w->dims[0]);
                st.K  = static_cast<std::size_t>(w->dims[1]);
                st.W  = wmap.at(nd.input[1]);
                if (nd.input.size() >= 3 && !nd.input[2].empty()) {
                    const Onnx::Initializer* b = graph.find(nd.input[2]);
                    if (!b) throw std::runtime_error("ForwardEngine: bad bias");
                    st.B = wmap.at(nd.input[2]);
                }
                const std::string& a = nd.input[0];
                if (elems.at(a) != st.K)
                    throw std::runtime_error("ForwardEngine: Gemm shape mismatch");
                elems[nd.output[0]] = st.M;
                st.in  = ref(a);
                st.out = ref(nd.output[0]);
                plan.push_back(st);
            } else if (nd.op_type == "Relu") {
                if (nd.input.empty() || nd.output.empty())
                    throw std::runtime_error("ForwardEngine: malformed Relu");
                Step st;
                st.op = Step::Relu;
                st.n  = elems.at(nd.input[0]);
                elems[nd.output[0]] = st.n;
                st.in  = ref(nd.input[0]);
                st.out = ref(nd.output[0]);
                plan.push_back(st);
            } else {
                throw std::runtime_error(
                    "ForwardEngine: unsupported op '" + nd.op_type + "'");
            }
        }

        in_size  = elems.at(gin);
        out_size = elems.at(gout);

        std::size_t acap = 0;
        for (std::size_t e : act_elems) acap += align_up(e * sizeof(float));
        act_bytes = acap;
        act_pool.emplace(acap ? acap : kAlign);
        act_ptr.assign(act_elems.size(), nullptr);
    }

    [[nodiscard]] float* resolve(const Ref& r,
                                 std::span<const float> in,
                                 std::span<float> out) noexcept {
        switch (r.kind) {
            case Ref::In:  return const_cast<float*>(in.data());
            case Ref::Out: return out.data();
            case Ref::Act: return act_ptr[r.slot];
        }
        return nullptr;  // unreachable
    }

    void run(std::span<const float> in, std::span<float> out,
             std::span<double> step_us) {
        if (in.size() != in_size)
            throw std::runtime_error("ForwardEngine::forward: input size");
        if (out.size() < out_size)
            throw std::runtime_error("ForwardEngine::forward: output too small");
        if (!step_us.empty() && step_us.size() != plan.size())
            throw std::runtime_error("ForwardEngine::forward_traced: step count");

        // Recycle the activation arena (O(1)) and bump-carve each slot.
        act_pool->reset();
        for (std::size_t s = 0; s < act_elems.size(); ++s) {
            act_ptr[s] = static_cast<float*>(
                act_pool->try_allocate(act_elems[s] * sizeof(float), kAlign));
            if (!act_ptr[s])
                throw std::runtime_error("ForwardEngine: activation arena full");
        }

        for (std::size_t index = 0; index < plan.size(); ++index) {
            const Step& st = plan[index];
            const auto start = step_us.empty()
                ? std::chrono::steady_clock::time_point{}
                : std::chrono::steady_clock::now();
            if (st.op == Step::Gemm) {
                const float* x = resolve(st.in, in, out);
                float*       y = resolve(st.out, in, out);
                Compute::matvec_f32(std::span<float>(y, st.M),
                                    st.W, x, st.M, st.K);
                if (st.B)
                    Compute::add_inplace(std::span<float>(y, st.M), st.B, st.M);
            } else {  // Relu
                const float* s = resolve(st.in, in, out);
                float*       d = resolve(st.out, in, out);
                Compute::relu(std::span<float>(d, st.n), s, st.n);
            }
            if (!step_us.empty()) {
                const auto finish = std::chrono::steady_clock::now();
                step_us[index] =
                    std::chrono::duration<double, std::micro>(finish - start).count();
            }
        }
    }

    void forward(std::span<const float> in, std::span<float> out) {
        run(in, out, {});
    }

    void forward_traced(std::span<const float> in, std::span<float> out,
                        std::span<double> step_us) {
        run(in, out, step_us);
    }
};

ForwardEngine::ForwardEngine(const std::filesystem::path& onnx_path)
    : p_(std::make_unique<Impl>(onnx_path)) {}

ForwardEngine::~ForwardEngine() = default;

void ForwardEngine::forward(std::span<const float> input,
                            std::span<float> output) {
    p_->forward(input, output);
}

std::size_t ForwardEngine::input_size()  const noexcept { return p_->in_size; }
std::size_t ForwardEngine::output_size() const noexcept { return p_->out_size; }
std::size_t ForwardEngine::num_steps()   const noexcept { return p_->plan.size(); }
std::size_t ForwardEngine::activation_bytes() const noexcept {
    return p_->act_bytes;
}

std::vector<ExecutionStep> ForwardEngine::execution_plan() const {
    std::vector<ExecutionStep> result;
    result.reserve(p_->plan.size());
    for (const Impl::Step& step : p_->plan) {
        if (step.op == Impl::Step::Gemm) {
            result.push_back({OperationKind::Gemm, step.K, step.M,
                              step.M, step.K, step.B != nullptr});
        } else {
            result.push_back({OperationKind::Relu, step.n, step.n,
                              0, 0, false});
        }
    }
    return result;
}

void ForwardEngine::forward_traced(std::span<const float> input,
                                   std::span<float> output,
                                   std::span<double> step_us) {
    p_->forward_traced(input, output, step_us);
}

} // namespace Strata::Engine
