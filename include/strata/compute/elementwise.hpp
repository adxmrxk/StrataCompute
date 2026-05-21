// =============================================================================
//  Strata::Compute — elementwise activation/bias operators
// -----------------------------------------------------------------------------
//  The small ops that sit between the heavy matvec kernels in a forward pass.
//  Same hard contract as the rest of Strata::Compute: noexcept, no dynamic
//  allocation, operate only on caller buffers. matvec dominates runtime, so
//  these are kept simple and trivially auto-vectorisable rather than
//  hand-intrinsic'd.
// =============================================================================
#ifndef STRATA_COMPUTE_ELEMENTWISE_HPP
#define STRATA_COMPUTE_ELEMENTWISE_HPP

#include <cstddef>
#include <span>

namespace Strata::Compute {

/// y[i] += b[i]   (Gemm bias / broadcast-add over a length-n vector).
void add_inplace(std::span<float> y, const float* b, std::size_t n) noexcept;

/// y[i] = max(y[i], 0)   (ReLU, in place).
void relu_inplace(std::span<float> y, std::size_t n) noexcept;

/// dst[i] = max(src[i], 0)   (ReLU into a distinct buffer).
void relu(std::span<float> dst, const float* src, std::size_t n) noexcept;

} // namespace Strata::Compute

#endif // STRATA_COMPUTE_ELEMENTWISE_HPP
