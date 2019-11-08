clear all
close all


eigen_include_path = '-I../../bazel-fbstab/external/eigen';

main_file = 'mex_test.cc';


mex(eigen_include_path,main_file);