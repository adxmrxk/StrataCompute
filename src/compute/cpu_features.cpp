// =============================================================================
//  Strata::Compute::cpu_features — CPUID-based detection (allocation-free)
// =============================================================================
#include "strata/compute/cpu_features.hpp"

#if defined(_MSC_VER)
#  include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#  include <cpuid.h>
#  include <immintrin.h>
#endif

#include <array>
#include <cstdint>

namespace Strata::Compute {
namespace {

struct Regs { std::uint32_t eax, ebx, ecx, edx; };

[[nodiscard]] Regs cpuid(std::uint32_t leaf, std::uint32_t subleaf) noexcept {
    Regs r{};
#if defined(_MSC_VER)
    std::array<int, 4> i{};
    __cpuidex(i.data(), static_cast<int>(leaf), static_cast<int>(subleaf));
    r = {static_cast<std::uint32_t>(i[0]), static_cast<std::uint32_t>(i[1]),
         static_cast<std::uint32_t>(i[2]), static_cast<std::uint32_t>(i[3])};
#elif defined(__GNUC__) || defined(__clang__)
    __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
#endif
    return r;
}

// Read XCR0 to confirm the OS actually saves YMM/ZMM state.
[[nodiscard]] std::uint64_t xcr0() noexcept {
#if defined(_MSC_VER)
    return _xgetbv(0);
#elif defined(__GNUC__) || defined(__clang__)
    std::uint32_t lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<std::uint64_t>(hi) << 32) | lo;
#else
    return 0;
#endif
}

[[nodiscard]] CpuFeatures detect() noexcept {
    CpuFeatures f;
    const Regs l0 = cpuid(0, 0);
    const std::uint32_t max_leaf = l0.eax;

    if (max_leaf >= 1) {
        const Regs l1 = cpuid(1, 0);
        f.sse2 = (l1.edx & (1u << 26)) != 0;
        const bool osxsave = (l1.ecx & (1u << 27)) != 0;
        const bool cpu_avx = (l1.ecx & (1u << 28)) != 0;
        f.fma = (l1.ecx & (1u << 12)) != 0;

        // YMM (bits 1:2) and ZMM (bits 5:7) OS-enablement.
        const std::uint64_t xc = osxsave ? xcr0() : 0;
        const bool ymm_os = (xc & 0x6) == 0x6;
        const bool zmm_os = (xc & 0xE6) == 0xE6;
        f.avx = cpu_avx && ymm_os;

        if (max_leaf >= 7) {
            const Regs l7 = cpuid(7, 0);
            f.avx2    = f.avx && ((l7.ebx & (1u << 5)) != 0);
            f.avx512f = zmm_os && ((l7.ebx & (1u << 16)) != 0);
            f.avx512_vnni = f.avx512f && ((l7.ecx & (1u << 11)) != 0);
            const Regs l7s1 = cpuid(7, 1);
            f.avx_vnni = f.avx2 && ((l7s1.eax & (1u << 4)) != 0);
        }
    }
    return f;
}

} // namespace

const CpuFeatures& cpu_features() noexcept {
    static const CpuFeatures cached = detect();  // thread-safe, one-time
    return cached;
}

} // namespace Strata::Compute
