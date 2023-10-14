#ifndef DEEPVAN_UTILS_MEMORY_H_
#define DEEPVAN_UTILS_MEMORY_H_

#include <memory>
#include <utility>

namespace deepvan {

void fast_memcpy(void *dst, const void *src, size_t len);
void neon_memcpy(float *dst, const float *src, size_t data_len);

namespace memory_internal {

// Traits to select proper overload and return type for `make_unique<>`.
template <typename T>
struct MakeUniqueResult {
  using scalar = std::unique_ptr<T>;
};
template <typename T>
struct MakeUniqueResult<T[]> {
  using array = std::unique_ptr<T[]>;
};
template <typename T, size_t N>
struct MakeUniqueResult<T[N]> {
  using invalid = void;
};

}  // namespace memory_internal

// gcc 4.8 has __cplusplus at 201301 but doesn't define make_unique.  Other
// supported compilers either just define __cplusplus as 201103 but have
// make_unique (msvc), or have make_unique whenever __cplusplus > 201103 (clang)
#if (__cplusplus > 201103L || defined(_MSC_VER)) && \
    !(defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ == 8)
using std::make_unique;
#else

// `make_unique` overload for non-array types.
template <typename T, typename... Args>
typename memory_internal::MakeUniqueResult<T>::scalar make_unique(
    Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// `make_unique` overload for an array T[] of unknown bounds.
// The array allocation needs to use the `new T[size]` form and cannot take
// element constructor arguments. The `std::unique_ptr` will manage destructing
// these array elements.
template <typename T>
typename memory_internal::MakeUniqueResult<T>::array make_unique(size_t n) {
  return std::unique_ptr<T>(new typename std::remove_extent<T>::type[n]());
}

// `make_unique` overload for an array T[N] of known bounds.
// This construction will be rejected.
template <typename T, typename... Args>
typename memory_internal::MakeUniqueResult<T>::invalid make_unique(
    Args&&... /* args */) = delete;
#endif
}  // namespace deepvan

#endif  // DEEPVAN_UTILS_MEMORY_H_
