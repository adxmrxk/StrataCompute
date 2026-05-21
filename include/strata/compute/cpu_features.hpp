// =============================================================================
//  Strata::Compute::CpuFeatures  —  runtime ISA detection
// -----------------------------------------------------------------------------
//  Detected once (lazy, allocation-free) and used by the kernel dispatcher to
//  pick the fastest correct implementation. The avx512* flags are the seam:
//  this dev CPU (i5-12400) reports them false, but deployment Xeons will
//  report them true and an AVX-512 kernel can slot in behind the dispatcher
//  without touching call sites.
// =============================================================================
#ifndef STRATA_COMPUTE_CPU_FEATURES_HPP
#define STRATA_COMPUTE_CPU_FEATURES_HPP

namespace Strata::Compute {

struct CpuFeatures {
    bool sse2        = false;
    bool avx         = false;
    bool avx2        = false;
    bool fma         = false;
    bool avx_vnni    = false;  // VEX-encoded 256-bit VNNI (Alder Lake+)
    bool avx512f     = false;  // reserved seam — no AVX-512 kernel yet
    bool avx512_vnni = false;
};

/// Cached CPU feature snapshot. First call runs CPUID; thereafter O(1).
[[nodiscard]] const CpuFeatures& cpu_features() noexcept;

} // namespace Strata::Compute

#endif // STRATA_COMPUTE_CPU_FEATURES_HPP
