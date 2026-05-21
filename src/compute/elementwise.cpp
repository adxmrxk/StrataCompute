// =============================================================================
//  Strata::Compute — elementwise operators (allocation-free, noexcept)
// =============================================================================
#include "strata/compute/elementwise.hpp"

namespace Strata::Compute {

void add_inplace(std::span<float> y, const float* b, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        y[i] += b[i];
    }
}

void relu_inplace(std::span<float> y, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        if (y[i] < 0.0f) y[i] = 0.0f;
    }
}

void relu(std::span<float> dst, const float* src, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = (src[i] > 0.0f) ? src[i] : 0.0f;
    }
}

} // namespace Strata::Compute
