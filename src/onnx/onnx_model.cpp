// =============================================================================
//  Strata::Onnx::OnnxModel — implementation (ONNX Runtime C++ API)
// =============================================================================
#include "strata/onnx/onnx_model.hpp"

#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace Strata::Onnx {
namespace {

// Product of a static shape, treating dynamic dims (<= 0) as a batch of 1.
std::size_t elem_count(const std::vector<std::int64_t>& shape) noexcept {
    std::size_t n = 1;
    for (std::int64_t d : shape) {
        n *= (d > 0) ? static_cast<std::size_t>(d) : std::size_t{1};
    }
    return n;
}

} // namespace

struct OnnxModel::Impl {
    Ort::Env            env;
    Ort::SessionOptions opts;
    Ort::Session        session{nullptr};
    Ort::AllocatorWithDefaultOptions alloc;

    std::vector<std::string>               in_names, out_names;
    std::vector<std::vector<std::int64_t>> in_shapes, out_shapes;

    Impl(const std::filesystem::path& path, int threads)
        : env(ORT_LOGGING_LEVEL_WARNING, "strata") {
        opts.SetIntraOpNumThreads(threads);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // path.c_str() is wchar_t* on Windows == ORTCHAR_T*, char* on POSIX.
        session = Ort::Session(env, path.c_str(), opts);

        const std::size_t ni = session.GetInputCount();
        const std::size_t no = session.GetOutputCount();
        in_names.reserve(ni);  in_shapes.reserve(ni);
        out_names.reserve(no); out_shapes.reserve(no);

        for (std::size_t i = 0; i < ni; ++i) {
            in_names.emplace_back(session.GetInputNameAllocated(i, alloc).get());
            in_shapes.push_back(session.GetInputTypeInfo(i)
                                    .GetTensorTypeAndShapeInfo()
                                    .GetShape());
        }
        for (std::size_t i = 0; i < no; ++i) {
            out_names.emplace_back(session.GetOutputNameAllocated(i, alloc).get());
            out_shapes.push_back(session.GetOutputTypeInfo(i)
                                     .GetTensorTypeAndShapeInfo()
                                     .GetShape());
        }
    }
};

OnnxModel::OnnxModel(const std::filesystem::path& model_path,
                     int intra_op_threads)
    : p_(std::make_unique<Impl>(model_path, intra_op_threads)) {}

OnnxModel::~OnnxModel() = default;

std::size_t OnnxModel::num_inputs()  const noexcept { return p_->in_names.size(); }
std::size_t OnnxModel::num_outputs() const noexcept { return p_->out_names.size(); }

const std::string& OnnxModel::input_name(std::size_t i) const {
    return p_->in_names.at(i);
}
const std::string& OnnxModel::output_name(std::size_t i) const {
    return p_->out_names.at(i);
}
const std::vector<std::int64_t>& OnnxModel::input_shape(std::size_t i) const {
    return p_->in_shapes.at(i);
}
const std::vector<std::int64_t>& OnnxModel::output_shape(std::size_t i) const {
    return p_->out_shapes.at(i);
}
std::size_t OnnxModel::input_elements(std::size_t i) const {
    return elem_count(p_->in_shapes.at(i));
}
std::size_t OnnxModel::output_elements(std::size_t i) const {
    return elem_count(p_->out_shapes.at(i));
}

void OnnxModel::run(std::span<const float> input, std::span<float> output) {
    if (p_->in_names.size() != 1 || p_->out_names.size() != 1) {
        throw std::runtime_error(
            "OnnxModel::run() supports single-input/single-output graphs");
    }
    if (input.size() != input_elements(0)) {
        throw std::runtime_error("OnnxModel::run(): input size mismatch");
    }
    if (output.size() < output_elements(0)) {
        throw std::runtime_error("OnnxModel::run(): output span too small");
    }

    // Concretise dynamic dims (-1) to batch 1 for the tensor descriptor.
    std::vector<std::int64_t> ishape = p_->in_shapes[0];
    for (std::int64_t& d : ishape) {
        if (d <= 0) d = 1;
    }

    const Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(input.data()), input.size(),
        ishape.data(), ishape.size());

    const char* in_n[]  = {p_->in_names[0].c_str()};
    const char* out_n[] = {p_->out_names[0].c_str()};

    std::vector<Ort::Value> outs = p_->session.Run(
        Ort::RunOptions{nullptr}, in_n, &in, 1, out_n, 1);

    const float* od = outs[0].GetTensorData<float>();
    const std::size_t n = outs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    for (std::size_t k = 0; k < n && k < output.size(); ++k) {
        output[k] = od[k];
    }
}

} // namespace Strata::Onnx
