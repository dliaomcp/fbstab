# Contributing to FBstab

## Pull Requests
We're not currently set up to accept outside pull requests, the core module is still under active development. If you'd like to contribute please contact us so we can discuss.

## Getting started
1. Fork the repo
2. Clone it to your local machine
   - `cd <path to repo>`
   - `git clone <url>` 
3. Add an "upstream" remote
   - `cd <path to repo>/reponame`
   - `git remote add upstream <url>`
   - `git remote set-url --push upstream no_push`

## Starting a new branch
1. `git fetch upstream`
2. `git checkout -b my_branch upstream/master`

## Preparing for a PR
1. Rebase to master
   - `git fetch upstream`
   - `git rebase -i upstream/master`

2. Push to your fork
   - `git push origin -f`

## Issues
We use GitHub issues to track public bugs.

## Repo Organization
FBstab uses [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) as its linear algebra library and is designed to be extensible for different classes of QPs. Its architecture is similar to that of [OOQP](http://pages.cs.wisc.edu/~swright/ooqp/).

The algorithm itself is implemented in an abstract way using template metaprogramming. It requires 5 component classes which each must implement several methods.

1. A Variable class for storing primal-dual variables.
2. A Residual class for computing optimality residuals.
3. A Data class for storing and manipulating problem data.
4. A LinearSolver class for computing Newton steps.
5. A Feasibility class for detecting primal or dual infeasibility.

These classes should be specialized for different kinds of QPs, e.g., support vector machines, optimal control problems, general dense QPs, in order to exploit structure. FBstab currently supports:

- Linear-quadratic optimal control problems with polyhedral constraints, i.e., model predictive control problems.
- Dense inequality constrained problems.

## Licensing
By contributing to FBstab, you agree that your contributions will be licensed under the LICENSE file in the root directory.

## Contributors
- Dominic Liao-McPherson (main developer)
- Soonho Kong
- Frank Permenter

