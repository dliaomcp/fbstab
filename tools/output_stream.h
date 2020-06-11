#pragma once

#include <cstdio>

#include "tools/copyable_macros.h"

namespace fbstab {

/**
 * The primary printing interface class. Its purpose is to enable FBstab to
 * print to a variety of output streams.
 *
 * @tparam T for the curiously recurring template pattern.
 */
template <class T>
class OutputStream {
 public:
  /**
   * Print to a output stream.
   * @param[in] message C-string to be printed
   */
  void Print(const char* message) const {
    static_cast<const T*>(this)->PrintImplementation(message);
  }
};

/**
 * Default class that prints to the standard output.
 */
class StandardOutput : public OutputStream<StandardOutput> {
 public:
  FBSTAB_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(StandardOutput)
  StandardOutput() = default;

 protected:
  void PrintImplementation(const char* message) const { printf("%s", message); }
  friend class OutputStream<StandardOutput>;
};

}  // namespace fbstab