#pragma once

// ref: https://forums.developer.nvidia.com/t/c-20s-source-location-compilation-error-when-using-nvcc-12-1/258026/3
#ifdef __CUDACC__
#include <cuda.h>
#if CUDA_VERSION <= 12010

#pragma push_macro("__cpp_consteval")
#pragma push_macro("_NODISCARD")
#pragma push_macro("__builtin_LINE")

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbuiltin-macro-redefined"
#define __cpp_consteval 201811L
#pragma clang diagnostic pop

#ifdef _NODISCARD
#undef _NODISCARD
#define _NODISCARD
#endif

#define consteval constexpr

#include "source_location.h"

#undef consteval
#pragma pop_macro("__cpp_consteval")
#pragma pop_macro("_NODISCARD")
#else
#include "source_location.h"
#endif
#else
#include "source_location.h"
#endif

#include <dlpack/dlpack.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace host {

template <typename>
struct dependent_false : std::false_type {};

template <typename T>
constexpr bool dependent_false_v = dependent_false<T>::value;

struct DebugInfo : public source_location_t {
  DebugInfo(source_location_t loc = source_location_t::current()) : source_location_t(loc) {}
};

struct PanicError : public std::runtime_error {
 public:
  explicit PanicError(std::string msg) : std::runtime_error(msg), m_message(std::move(msg)) {}

  std::string_view root_cause() const {
    const std::string_view str{m_message};
    const auto pos = str.find(": ");
    return pos == std::string_view::npos ? str : str.substr(pos + 2);
  }

 private:
  std::string m_message;
};

template <typename... Args>
[[noreturn]] inline void panic(DebugInfo location, Args&&... args) {
  std::ostringstream os;
  os << "Runtime check failed at " << location.file_name() << ":" << location.line();
  if constexpr (sizeof...(args) > 0) {
    os << ": ";
    using swallow = int[];
    (void)swallow{0, ((void)(os << std::forward<Args>(args)), 0)...};
  } else {
    os << " in " << location.function_name();
  }
  throw PanicError(std::move(os).str());
}

template <typename... Args>
struct RuntimeCheck {
  template <typename Cond>
  explicit RuntimeCheck(Cond&& condition, Args&&... args, DebugInfo location = DebugInfo()) {
    if (condition) return;
    ::host::panic(location, std::forward<Args>(args)...);
  }

  template <typename Cond>
  explicit RuntimeCheck(DebugInfo location, Cond&& condition, Args&&... args) {
    if (condition) return;
    ::host::panic(location, std::forward<Args>(args)...);
  }
};

template <typename... Args>
struct Panic {
  explicit Panic(Args&&... args, DebugInfo location = DebugInfo()) {
    ::host::panic(location, std::forward<Args>(args)...);
  }

  explicit Panic(DebugInfo location, Args&&... args) {
    ::host::panic(location, std::forward<Args>(args)...);
  }

  ~Panic() noexcept {
    std::terminate();
  }
};

template <typename Cond, typename... Args>
RuntimeCheck(Cond&&, Args&&...) -> RuntimeCheck<Args...>;

template <typename Cond, typename... Args>
RuntimeCheck(DebugInfo, Cond&&, Args&&...) -> RuntimeCheck<Args...>;

template <typename... Args>
Panic(Args&&...) -> Panic<Args...>;

template <typename... Args>
Panic(DebugInfo, Args&&...) -> Panic<Args...>;

namespace pointer {

template <typename... U>
inline std::ptrdiff_t sum_offsets(U... values) {
  return (std::ptrdiff_t(0) + ... + static_cast<std::ptrdiff_t>(values));
}

// we only allow void * pointer arithmetic for safety
template <
    typename T = char,
    typename... U,
    typename std::enable_if<std::conjunction<std::is_integral<U>...>::value, int>::type = 0>
inline void* offset(void* ptr, U... offset_values) {
  return static_cast<T*>(ptr) + sum_offsets(offset_values...);
}

template <
    typename T = char,
    typename... U,
    typename std::enable_if<std::conjunction<std::is_integral<U>...>::value, int>::type = 0>
inline const void* offset(const void* ptr, U... offset_values) {
  return static_cast<const T*>(ptr) + sum_offsets(offset_values...);
}

}  // namespace pointer

template <
    typename T,
    typename U,
    typename std::enable_if<std::is_integral<T>::value && std::is_integral<U>::value, int>::type = 0>
inline constexpr auto div_ceil(T a, U b) -> decltype((a + b - 1) / b) {
  return (a + b - 1) / b;
}

template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline constexpr bool has_single_bit(T value) {
  using U = typename std::make_unsigned<T>::type;
  const auto unsigned_value = static_cast<U>(value);
  return unsigned_value != 0 && (unsigned_value & (unsigned_value - 1)) == 0;
}

template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline constexpr int32_t countr_zero(T value) {
  using U = typename std::make_unsigned<T>::type;
  auto unsigned_value = static_cast<U>(value);
  if (unsigned_value == 0) {
    return static_cast<int32_t>(sizeof(U) * 8);
  }

  int32_t count = 0;
  while ((unsigned_value & U{1}) == 0) {
    unsigned_value >>= 1;
    ++count;
  }
  return count;
}

inline std::size_t dtype_bytes(DLDataType dtype) {
  return static_cast<std::size_t>(dtype.bits / 8);
}

template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
class irange_iterator {
 public:
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::input_iterator_tag;
  using pointer = const T*;
  using reference = T;

  explicit irange_iterator(T value) : value_(value) {}

  T operator*() const { return value_; }

  irange_iterator& operator++() {
    ++value_;
    return *this;
  }

  irange_iterator operator++(int) {
    irange_iterator tmp(*this);
    ++(*this);
    return tmp;
  }

  bool operator!=(const irange_iterator& other) const { return value_ != other.value_; }
  bool operator==(const irange_iterator& other) const { return value_ == other.value_; }

 private:
  T value_;
};

template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
class irange_view {
 public:
  irange_view(T begin, T end) : begin_(begin), end_(end) {}

  irange_iterator<T> begin() const { return irange_iterator<T>(begin_); }
  irange_iterator<T> end() const { return irange_iterator<T>(end_); }

 private:
  T begin_;
  T end_;
};

template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline irange_view<T> irange(T end) {
  return irange_view<T>(static_cast<T>(0), end);
}

template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline irange_view<T> irange(T start, T end) {
  return irange_view<T>(start, end);
}

}  // namespace host
