# -*- python -*-

# This file marks a workspace root for the Bazel build system. see
# https://bazel.build/ .

workspace(name = "fbstab")

# External dependencies are added here, see the various folders under
# fbstab/tools/ for definitions.
load("@fbstab//tools/eigen:repository.bzl","eigen_repository")
load("@fbstab//tools/gtest:repository.bzl","gtest_repository")

# By default fbstab downloads an appropriate version of Eigen.
# If you'd like to use a version installed on your system instead
# set use_local = True and supply the local absolute path.
# This should contain the Eigen/ directory that contains 
# Eigen, Dense, Core etc. (not the one that contains e.g., INSTALL)

eigen_repository(name = "eigen", use_local = False, local_path = "/usr/local/include/")
gtest_repository(name = "gtest")
