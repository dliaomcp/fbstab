# -*- python -*-

# This file marks a workspace root for the Bazel build system. see
# https://bazel.build/ .

workspace(name = "fbstab")

load("@fbstab//tools/eigen:repository.bzl","eigen_repository")
load("@fbstab//tools/gtest:repository.bzl","gtest_repository")

eigen_repository(name = "eigen")
gtest_repository(name = "gtest")
