#pragma once
// Based on the drake macros
// https://github.com/RobotLocomotion/drake/blob/master/common/drake_copyable.h

/**
Provides careful macros to selectively enable or disable the special member
functions for copy-construction, copy-assignment, move-construction, and
move-assignment.

http://en.cppreference.com/w/cpp/language/member_functions#Special_member_functions

When enabled via these macros, the `= default` implementation is provided.
Code that needs custom copy or move functions should not use these macros.
*/

#define FBSTAB_NO_COPY_NO_MOVE_NO_ASSIGN(Classname) \
  Classname(const Classname &) = delete;            \
  void operator=(const Classname &) = delete;       \
  Classname(Classname &&) = delete;                 \
  void operator=(Classname &&) = delete;

#define FBSTAB_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Classname)           \
  Classname(const Classname &) = default;                            \
  Classname &operator=(const Classname &) = default;                 \
  Classname(Classname &&) = default;                                 \
  Classname &operator=(Classname &&) = default;                      \
  /* Fails at compile-time if default-copy doesn't work. */          \
  static void FBSTAB_COPYABLE_DEMAND_COPY_CAN_COMPILE() {            \
    (void)static_cast<Classname &(Classname::*)(const Classname &)>( \
        &Classname::operator=);                                      \
  }
