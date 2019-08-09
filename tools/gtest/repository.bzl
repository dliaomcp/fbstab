# -*- python -*-

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def gtest_repository(name):
    git_repository(
        name = name,
        remote = "https://github.com/google/googletest.git",
        commit = "2fe3bd994b3189899d93f1d5a881e725e046fdc2",
        shallow_since = "1535728917 -0400",
    )
