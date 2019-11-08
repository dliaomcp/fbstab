clear all
close all



% Create a cell array of matrices
a = rand(3,2);
b = rand(2,3);
A = {a,b};


s.A = {a,a};
s.B = {b,b};
% call the mex function
mex_test(A,s);