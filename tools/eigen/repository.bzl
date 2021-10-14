# -*- python -*-
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def eigen_repository(name,use_local = False, local_path = "/usr/local/include/"):
    if not use_local:
        http_archive(
          name = name,
          urls = ["https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"],
          sha256 = "1ccaabbfe870f60af3d6a519c53e09f3dcf630207321dffa553564a8e75c4fc8",
          build_file = "//tools/eigen:package.BUILD.bazel",
          strip_prefix = "eigen-3.4.0",
        )
    else:
        native.new_local_repository(
        name = name,
        path = local_path,
        build_file = "//tools/eigen:package.BUILD.bazel"
        )

    

