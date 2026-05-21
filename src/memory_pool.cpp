// =============================================================================
//  Strata::MemoryPool — implementation
// =============================================================================
#include "strata/memory_pool.hpp"

#include <cassert>
#include <cstring>
#include <utility>

namespace Strata {
namespace {

[[nodiscard]] constexpr bool is_pow2(std::size_t v) noexcept {
    return v != 0 && (v & (v - 1)) == 0;
}

/// Round `n` up to the next multiple of `align` (align must be a power of 2).
[[nodiscard]] constexpr std::size_t align_up(std::size_t n,
                                             std::size_t align) noexcept {
    return (n + (align - 1)) & ~(align - 1);
}

} // namespace

MemoryPool::MemoryPool(std::size_t capacity_bytes, std::size_t base_alignment)
    : capacity_(capacity_bytes), base_align_(base_alignment) {
    assert(is_pow2(base_alignment) && "base_alignment must be a power of two");
    assert(capacity_bytes > 0 && "arena capacity must be non-zero");

    // The single, deliberate heap reservation for the whole pool's lifetime.
    // C++17 aligned operator new guarantees the requested over-alignment.
    base_ = static_cast<std::byte*>(
        ::operator new(capacity_bytes, std::align_val_t{base_alignment}));
}

MemoryPool::~MemoryPool() {
    if (base_ != nullptr) {
        ::operator delete(base_, std::align_val_t{base_align_});
    }
}

MemoryPool::MemoryPool(MemoryPool&& o) noexcept
    : base_(o.base_), capacity_(o.capacity_), offset_(o.offset_),
      high_water_(o.high_water_), base_align_(o.base_align_) {
    o.base_ = nullptr;
    o.capacity_ = o.offset_ = o.high_water_ = 0;
}

MemoryPool& MemoryPool::operator=(MemoryPool&& o) noexcept {
    if (this != &o) {
        if (base_ != nullptr) {
            ::operator delete(base_, std::align_val_t{base_align_});
        }
        base_       = o.base_;
        capacity_   = o.capacity_;
        offset_     = o.offset_;
        high_water_ = o.high_water_;
        base_align_ = o.base_align_;
        o.base_ = nullptr;
        o.capacity_ = o.offset_ = o.high_water_ = 0;
    }
    return *this;
}

void* MemoryPool::try_allocate(std::size_t bytes,
                               std::size_t alignment) noexcept {
    assert(is_pow2(alignment) && "alignment must be a power of two");
    // Sub-base alignment is always satisfiable; alignment must not exceed the
    // arena base alignment or aligned blocks could fall outside guarantees.
    assert(alignment <= base_align_ &&
           "requested alignment exceeds arena base alignment");

    const std::size_t aligned = align_up(offset_, alignment);
    // Overflow-safe capacity check (aligned + bytes could wrap on huge inputs).
    if (aligned > capacity_ || bytes > capacity_ - aligned) {
        return nullptr;
    }

    void* p  = base_ + aligned;
    offset_  = aligned + bytes;
    if (offset_ > high_water_) {
        high_water_ = offset_;
    }
    return p;
}

void* MemoryPool::allocate(std::size_t bytes, std::size_t alignment) {
    void* p = try_allocate(bytes, alignment);
    if (p == nullptr) {
        throw std::bad_alloc{};
    }
    return p;
}

void MemoryPool::reset(bool zero_fill) noexcept {
    if (zero_fill && base_ != nullptr && offset_ > 0) {
        std::memset(base_, 0, offset_);
    }
    offset_ = 0;  // recycle the arena; high_water_ is intentionally retained
}

} // namespace Strata
