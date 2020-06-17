#pragma once

#include <algorithm>

namespace fbstab {
namespace tools {

template <class T>
struct _Unique_if {
  typedef std::unique_ptr<T> _Single_object;
};

template <class T>
struct _Unique_if<T[]> {
  typedef std::unique_ptr<T[]> _Unknown_bound;
};

template <class T, size_t N>
struct _Unique_if<T[N]> {
  typedef void _Known_bound;
};

/**
 *  std::make_unique replacement for C++11
 *  @param args the arguments to make_unique (forwarded)
 *  @return the unique pointer
 */
template <class T, class... Args>
typename _Unique_if<T>::_Single_object make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <class T>
typename _Unique_if<T>::_Unknown_bound make_unique(size_t n) {
  typedef typename std::remove_extent<T>::type U;
  return std::unique_ptr<T>(new U[n]());
}

template <class T, class... Args>
typename _Unique_if<T>::_Known_bound make_unique(Args&&...) = delete;

// Projects x onto [a,b].
template <class T>
T saturate(const T& x, const T& a, const T& b) {
  if (a > b) {
    throw std::runtime_error(
        "In tools::saturate: upper bound must be larger than the "
        "lower bound");
  }
  const T temp = std::min(x, b);
  return std::max(temp, a);
}

}  // namespace tools
}  // namespace fbstab
