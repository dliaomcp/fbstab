# Based on Drake's drake.bzl file,
# https://github.com/RobotLocomotion/drake/blob/master/tools/drake.bzl.

# The CXX_FLAGS will be enabled for all C++ rules in the project
# building with any compiler.
CXX_FLAGS = [
    "-Werror=all",
    "-Werror=attributes",
    "-Werror=deprecated",
    "-Werror=deprecated-declarations",
    "-Werror=extra",
    "-Werror=ignored-qualifiers",
    "-Werror=old-style-cast",
    "-Werror=overloaded-virtual",
    "-Werror=pedantic",
    "-Werror=shadow",
]


def _platform_copts(rule_copts, cc_test = 0):
    return CXX_FLAGS + rule_copts

def fbstab_cc_library(
        name,
        hdrs = None,
        srcs = None,
        deps = None,
        copts = [],
        **kwargs):
    """Creates a rule to declare a C++ library."""
    native.cc_library(
        name = name,
        hdrs = hdrs,
        srcs = srcs,
        deps = deps,
        copts = _platform_copts(copts),
        **kwargs
    )

def fbstab_cc_binary(
        name,
        hdrs = None,
        srcs = None,
        deps = None,
        copts = [],
        **kwargs):
    """Creates a rule to declare a C++ binary."""
    native.cc_binary(
        name = name,
        hdrs = hdrs,
        srcs = srcs,
        deps = deps,
        copts = _platform_copts(copts),
        **kwargs
    )

def fbstab_cc_googletest(
        name,
        srcs,
        deps = None,
        size = None,
        copts = [],
        use_default_main = True,
        **kwargs):
    """Creates a rule to declare a C++ unit test using googletest.
    Always adds a deps= entry for googletest main
    (@com_google_googletest//:gtest_main).
    """
    if size == None:
        size = "small"
    if deps == None:
        deps = []
    if use_default_main:
        deps.append("@gtest//:gtest_main")
    else:
        deps.append("@gtest//:gtest")
    native.cc_test(
        name = name,
        size = size,
        srcs = srcs,
        deps = deps,
        copts = _platform_copts(copts),
        **kwargs
    )
