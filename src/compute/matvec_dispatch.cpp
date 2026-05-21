// =============================================================================
//  Strata::Compute — kernel dispatcher
// -----------------------------------------------------------------------------
//  Resolves Backend::Auto against cpu_features() and routes the dispatched
//  entry points to a pinned kernel. The selection override is a lock-free
//  atomic (no allocation, thread-safe) so benchmarks/tests can pin Scalar vs
//  AVX2. Backend::AVX512 is the reserved seam: until an AVX-512 kernel exists
//  it transparently degrades to AVX2 (or Scalar) so call sites never change.
// =============================================================================
#include "strata/compute/matvec.hpp"
#include "strata/compute/cpu_features.hpp"

#include <atomic>
#include <cassert>

namespace Strata::Compute {
namespace {

std::atomic<Backend> g_selection{Backend::Auto};

/// Collapse Auto/AVX512 to a concrete, currently-implemented kernel.
[[nodiscard]] Backend resolve(Backend sel) noexcept {
    const CpuFeatures& f = cpu_features();
    const bool have_avx2 = f.avx2 && f.fma;
    switch (sel) {
        case Backend::Scalar: return Backend::Scalar;
        case Backend::AVX2:   return have_avx2 ? Backend::AVX2 : Backend::Scalar;
        case Backend::AVX512: // seam: no kernel yet -> best available
        case Backend::Auto:
        default:              return have_avx2 ? Backend::AVX2 : Backend::Scalar;
    }
}

} // namespace

void set_backend(Backend b) noexcept {
    g_selection.store(b, std::memory_order_relaxed);
}

Backend active_backend() noexcept {
    return resolve(g_selection.load(std::memory_order_relaxed));
}

void matvec_f32(std::span<float> y, const float* W, const float* x,
                std::size_t M, std::size_t K) noexcept {
    assert(y.size() >= M && "matvec_f32: output span smaller than M");
    if (resolve(g_selection.load(std::memory_order_relaxed)) == Backend::AVX2) {
        kernels::matvec_f32_avx2(y.data(), W, x, M, K);
    } else {
        kernels::matvec_f32_scalar(y.data(), W, x, M, K);
    }
}

void matvec_i8(std::span<float> y, const std::int8_t* W, const std::int8_t* x,
               std::size_t M, std::size_t K, float scale) noexcept {
    assert(y.size() >= M && "matvec_i8: output span smaller than M");
    if (resolve(g_selection.load(std::memory_order_relaxed)) == Backend::AVX2) {
        kernels::matvec_i8_avx2(y.data(), W, x, M, K, scale);
    } else {
        kernels::matvec_i8_scalar(y.data(), W, x, M, K, scale);
    }
}

} // namespace Strata::Compute
