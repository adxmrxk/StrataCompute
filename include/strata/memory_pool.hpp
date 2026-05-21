// =============================================================================
//  Strata::MemoryPool  —  zero-copy arena / bump allocator
// -----------------------------------------------------------------------------
//  Purpose
//  -------
//  Eliminates *all* runtime heap traffic (malloc/free, new/delete) on the
//  model forward-pass hot path. A single contiguous, cache-line-aligned arena
//  is reserved once at initialisation (sized to the network's maximum
//  activation-graph footprint). During inference, activation tensors are
//  carved out with an O(1) pointer bump and the whole arena is recycled with
//  an O(1) `reset()` between inferences.
//
//  Hot-path contract
//  -----------------
//   * The constructor performs the *only* heap allocation.
//   * `try_allocate()` is `noexcept` and never touches the heap — this is the
//     call any code under the `Strata::Compute` namespace must use.
//   * `allocate()` is a convenience that throws `std::bad_alloc` on arena
//     exhaustion; intended for setup/wiring, not the steady-state hot path.
//   * `reset()` is `noexcept`, O(1), and performs no deallocation.
//
//  Not thread-safe by design: inference threads each own a private arena,
//  which is exactly what removes synchronisation from the hot path.
// =============================================================================
#ifndef STRATA_MEMORY_POOL_HPP
#define STRATA_MEMORY_POOL_HPP

#include <cstddef>
#include <new>

namespace Strata {

/// Default arena/base alignment: one modern x86 cache line (64 bytes).
inline constexpr std::size_t kCacheLineBytes = 64;

class MemoryPool {
public:
    /// Reserve a contiguous arena of `capacity_bytes`, with the arena base
    /// aligned to `base_alignment` (must be a power of two, >= alignof(max)).
    /// Throws std::bad_alloc if the backing reservation fails.
    explicit MemoryPool(std::size_t capacity_bytes,
                         std::size_t base_alignment = kCacheLineBytes);

    ~MemoryPool();

    MemoryPool(const MemoryPool&)            = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&& other) noexcept;
    MemoryPool& operator=(MemoryPool&& other) noexcept;

    /// Hot-path-safe carve. Aligns the bump cursor up to `alignment`
    /// (power of two) and returns the block, or nullptr if the arena cannot
    /// satisfy the request. Never throws, never hits the heap.
    [[nodiscard]] void* try_allocate(std::size_t bytes,
                                     std::size_t alignment) noexcept;

    /// Setup-time carve. Same as try_allocate() but throws std::bad_alloc on
    /// exhaustion. Do not call on the inference hot path.
    [[nodiscard]] void* allocate(std::size_t bytes,
                                 std::size_t alignment = alignof(std::max_align_t));

    /// Recycle the whole arena. O(1); no memory is returned to the OS.
    /// If `zero_fill` is true the used region is scrubbed (debug aid; off the
    /// hot path because it is O(used)).
    void reset(bool zero_fill = false) noexcept;

    [[nodiscard]] std::size_t capacity()        const noexcept { return capacity_; }
    [[nodiscard]] std::size_t used()            const noexcept { return offset_; }
    [[nodiscard]] std::size_t remaining()       const noexcept { return capacity_ - offset_; }
    /// Largest `used()` ever reached since construction — size your arena to this.
    [[nodiscard]] std::size_t high_water_mark() const noexcept { return high_water_; }
    [[nodiscard]] const void* data()            const noexcept { return base_; }

private:
    std::byte*  base_       = nullptr;  // arena origin (base_alignment-aligned)
    std::size_t capacity_   = 0;        // total reserved bytes
    std::size_t offset_     = 0;        // bump cursor (bytes from base_)
    std::size_t high_water_ = 0;        // peak offset_ observed
    std::size_t base_align_ = kCacheLineBytes;
};

} // namespace Strata

#endif // STRATA_MEMORY_POOL_HPP
