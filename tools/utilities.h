#pragma once

#include <algorithm>

namespace fbstab {
namespace tools {

/**
 *  std::make_unique replacement for C++11
 *  @param args the arguments to make_unique (forwarded)
 *  @return the unique pointer
 */
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

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