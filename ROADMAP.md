# Roadmap

## Core algorithm
- Add adaptive regularization parameter updating
- Add variable metric preconditioning
- Refactor semismooth subproblem solver to use a free slack variable
- Add interior point subproblem solver

## Component classes
- Write general sparse matrix components
- Add bound support to MPC
- Add support for differently dimensioned terminal and initial constraints to MPC
- Write sparse MPC components

## Interfaces
- MATLAB interface
- Python bindings

## Code and such
- Write abstract interface class for components
- Setup travis-ci
- Update option interface
- Add use_blas option (through Eigen)
- Add dynamic memory checking functionality to tests
- Add apple_debug config that allows stack tracing
- Add CMake build system
