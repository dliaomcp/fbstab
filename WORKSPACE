# -*- python -*-

# This file marks a workspace root for the Bazel build system. see
# https://bazel.build/ .

workspace(name = "fbstab")

# External dependencies are added here, see the various folders under
# fbstab/tools/ for definitions.
load("@fbstab//tools/eigen:repository.bzl","eigen_repository")
load("@fbstab//tools/gtest:repository.bzl","gtest_repository")

eigen_repository(name = "eigen")
gtest_repository(name = "gtest")
