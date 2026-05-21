// =============================================================================
//  Strata::Tensor<T>  —  cache-aligned, zero-copy multidimensional view
// -----------------------------------------------------------------------------
//  A Tensor is a *non-owning* row-major view onto memory carved from a
//  Strata::MemoryPool. It never allocates: shape/stride metadata lives in
//  fixed-size storage (no std::vector), and element storage is borrowed from
//  the arena. The arena base is 64-byte aligned, so every Tensor created via
//  `create()` starts on a cache line — enabling aligned SIMD loads and
//  avoiding false sharing / cache-line splits across rows.
//
//  Design contract
//  ---------------
//   * Rank is bounded by kMaxRank so metadata is stack/inline (alloc-free).
//   * Strides are row-major (C order); the innermost stride is 1 element.
//   * `flat()` / `row()` expose std::span views (C++20) for safe iteration.
//   * `prefetch_row()` issues an explicit _mm_prefetch (portable fallback)
//     to stage upcoming rows into L1/L2 before the math kernel touches them.
// =============================================================================
#ifndef STRATA_TENSOR_HPP
#define STRATA_TENSOR_HPP

#include "strata/memory_pool.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <span>
#include <type_traits>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#  include <immintrin.h>
#  define STRATA_HAS_MM_PREFETCH 1
#else
#  define STRATA_HAS_MM_PREFETCH 0
#endif

namespace Strata {

/// Maximum tensor rank — keeps shape/stride metadata allocation-free.
inline constexpr std::size_t kMaxRank = 8;

/// Element types StrataCompute computes on (extended with INT8/FP16 in Phase 2).
template <class T>
concept TensorElement = std::is_arithmetic_v<T>;

/// Cache locality hint for prefetch(), mapped to the _MM_HINT_* family.
enum class Locality : int {
    NonTemporal = 0,  // _MM_HINT_NTA — stream, do not pollute cache
    L3          = 1,  // _MM_HINT_T2
    L2          = 2,  // _MM_HINT_T1
    L1          = 3   // _MM_HINT_T0 — bring all the way into L1
};

template <TensorElement T>
class Tensor {
public:
    using value_type = T;

    Tensor() = default;

    /// Carve a 64-byte-aligned, contiguous row-major tensor from `pool`.
    /// Throws std::bad_alloc only if the arena is exhausted (setup-time).
    [[nodiscard]] static Tensor create(MemoryPool& pool,
                                       std::initializer_list<std::size_t> shape) {
        Tensor t;
        t.set_shape(shape);
        void* p = pool.allocate(t.nbytes(), kCacheLineBytes);
        t.data_ = static_cast<T*>(p);
        return t;
    }

    /// Wrap pre-existing aligned storage (e.g. mmapped weights). The caller
    /// guarantees `data` points to at least nbytes() of T-aligned memory.
    [[nodiscard]] static Tensor wrap(T* data,
                                     std::initializer_list<std::size_t> shape) {
        Tensor t;
        t.set_shape(shape);
        t.data_ = data;
        return t;
    }

    // ---- Shape / stride introspection -------------------------------------
    [[nodiscard]] std::size_t rank() const noexcept { return rank_; }

    [[nodiscard]] std::size_t shape(std::size_t dim) const noexcept {
        assert(dim < rank_ && "shape() dim out of range");
        return shape_[dim];
    }

    [[nodiscard]] std::size_t stride(std::size_t dim) const noexcept {
        assert(dim < rank_ && "stride() dim out of range");
        return stride_[dim];
    }

    /// Total element count (product of the shape).
    [[nodiscard]] std::size_t size() const noexcept { return count_; }

    [[nodiscard]] std::size_t nbytes() const noexcept {
        return count_ * sizeof(T);
    }

    [[nodiscard]] bool empty() const noexcept { return count_ == 0; }

    [[nodiscard]] T*       data()       noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }

    // ---- Span views (C++20) ----------------------------------------------
    [[nodiscard]] std::span<T> flat() noexcept {
        return {data_, count_};
    }
    [[nodiscard]] std::span<const T> flat() const noexcept {
        return {data_, count_};
    }

    /// Row span for a rank>=1 tensor: the contiguous innermost run reached by
    /// fixing every leading index. `outer` indexes the leading dimension.
    [[nodiscard]] std::span<T> row(std::size_t outer) noexcept {
        assert(rank_ >= 1 && outer < shape_[0] && "row() out of range");
        const std::size_t len = (rank_ == 1) ? 1 : stride_[0];
        return {data_ + outer * stride_[0], len};
    }

    // ---- Element access ---------------------------------------------------
    /// Row-major flat offset for the given multi-index.
    template <class... Idx>
    [[nodiscard]] std::size_t index(Idx... idx) const noexcept {
        static_assert(sizeof...(Idx) <= kMaxRank, "too many indices");
        assert(sizeof...(Idx) == rank_ && "index() rank mismatch");
        const std::array<std::size_t, sizeof...(Idx)> c{
            static_cast<std::size_t>(idx)...};
        std::size_t off = 0;
        for (std::size_t d = 0; d < c.size(); ++d) {
            assert(c[d] < shape_[d] && "index out of bounds");
            off += c[d] * stride_[d];
        }
        return off;
    }

    template <class... Idx>
    [[nodiscard]] T& operator()(Idx... idx) noexcept {
        return data_[index(idx...)];
    }
    template <class... Idx>
    [[nodiscard]] const T& operator()(Idx... idx) const noexcept {
        return data_[index(idx...)];
    }

    // ---- Cache prefetch ---------------------------------------------------
    /// Stage `data_ + elem_offset` into the cache level named by `loc`.
    static void prefetch(const T* addr, Locality loc = Locality::L1) noexcept {
#if STRATA_HAS_MM_PREFETCH
        switch (loc) {
            case Locality::NonTemporal:
                _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_NTA); break;
            case Locality::L3:
                _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T2); break;
            case Locality::L2:
                _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T1); break;
            case Locality::L1:
                _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0); break;
        }
#elif defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(addr, 0 /*read*/,
                           (loc == Locality::NonTemporal) ? 0 : 3);
#else
        (void)addr; (void)loc;
#endif
    }

    /// Prefetch the start of a leading-dimension row before kernel use.
    void prefetch_row(std::size_t outer,
                      Locality loc = Locality::L1) const noexcept {
        assert(rank_ >= 1 && outer < shape_[0] && "prefetch_row() out of range");
        prefetch(data_ + outer * stride_[0], loc);
    }

    /// True iff the data pointer sits on a 64-byte cache-line boundary.
    [[nodiscard]] bool is_cache_aligned() const noexcept {
        return (reinterpret_cast<std::uintptr_t>(data_) % kCacheLineBytes) == 0;
    }

private:
    void set_shape(std::initializer_list<std::size_t> shape) noexcept {
        assert(shape.size() >= 1 && shape.size() <= kMaxRank &&
               "tensor rank must be in [1, kMaxRank]");
        rank_ = shape.size();
        std::size_t d = 0;
        for (std::size_t s : shape) {
            assert(s > 0 && "tensor dimensions must be non-zero");
            shape_[d++] = s;
        }
        // Row-major strides: innermost == 1, then running product.
        stride_[rank_ - 1] = 1;
        for (std::size_t i = rank_ - 1; i-- > 0;) {
            stride_[i] = stride_[i + 1] * shape_[i + 1];
        }
        count_ = 1;
        for (std::size_t i = 0; i < rank_; ++i) count_ *= shape_[i];
    }

    T*                                  data_  = nullptr;
    std::array<std::size_t, kMaxRank>   shape_{};
    std::array<std::size_t, kMaxRank>   stride_{};
    std::size_t                         rank_  = 0;
    std::size_t                         count_ = 0;
};

} // namespace Strata

#endif // STRATA_TENSOR_HPP
