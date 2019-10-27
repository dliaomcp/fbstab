#pragma once

/**
 *  std::make_unique replacement for C++11
 *  @param args the arguments to make_unique (forwarded)
 *  @return the unique pointer
 */
namespace fbstab {
namespace tools {

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}  // namespace tools
}  // namespace fbstab