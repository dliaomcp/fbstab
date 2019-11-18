# -*- python -*-
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def eigen_repository(name,use_local = False, local_path = "/usr/local/include/"):
    if not use_local:
        http_archive(
          name = name,
          urls = ["http://bitbucket.org/eigen/eigen/get/3.3.7.zip"],
          sha256 = "65d3aebb5094280869955bcfb41aada2f5194e2d608f930951e810ce4c945c0b",
          build_file = "//tools/eigen:package.BUILD.bazel",
          strip_prefix = "eigen-eigen-323c052e1731",
        )
    else:
        native.new_local_repository(
        name = name,
        path = local_path,
        build_file = "//tools/eigen:package.BUILD.bazel"
        )

    

