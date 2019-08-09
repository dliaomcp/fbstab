# -*- python -*-

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def eigen_repository(name):
    new_git_repository(
        name = name,
        remote = "https://github.com/eigenteam/eigen-git-mirror.git",
        commit = "9e97af7de76716c99abdbfd4a4acb182ef098808",
        shallow_since = "1487684194 +0100",
        build_file = "//tools/eigen:package.BUILD.bazel",
    )
