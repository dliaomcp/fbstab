clear all
close all


include_paths = {'-I../../bazel-fbstab/external/eigen', '-I../../'};

main_file = 'fbstab_dense_mex.cc';


% mex('-v',include_paths{:},main_file);
mex(include_paths{:},main_file);