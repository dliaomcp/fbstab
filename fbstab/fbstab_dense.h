#pragma once

#include <Eigen/Dense>
#include <memory>

#include "fbstab/components/dense_cholesky_solver.h"
#include "fbstab/components/dense_data.h"
#include "fbstab/components/full_feasibility.h"
#include "fbstab/components/full_residual.h"
#include "fbstab/components/full_variable.h"
#include "fbstab/fbstab_algorithm.h"
#include "tools/copyable_macros.h"

namespace fbstab {

/** Convenience typedef for the templated dense version of the algorithm. */
using FBstabAlgoDense = FBstabAlgorithm<FullVariable, FullResidual, DenseData,
                                        DenseCholeskySolver, FullFeasibility>;

/**
 * FBstabDense implements the Proximally Stabilized Semismooth Algorithm
 * for solving convex quadratic programs of the following form (1):
 *
 *     min.    1/2  z'Hz + f'z
 *     s.t.         Az <= b
 *
 * where H is symmetric and positive semidefinite and its dual
 *
 *     min.   1/2  z'Hz + b'v
 *     s.t.   Hz + f + A'v = 0
 *            v >= 0.
 *
 * Or equivalently for solving its KKT system
 *
 *     Hz + f + A' v = 0
 *     Az <= b, v >= 0
 *     (b - Az)' v = 0
 *
 * where v is a dual variable.
 *
 * The algorithm is described in https://arxiv.org/pdf/1901.04046.pdf.
 * Aside from convexity there are no assumptions made about the problem.
 * This method can detect unboundedness/infeasibility and accepts
 * arbitrary initial guesses.
 *
 * The problem is of size (nz,nv) where:
 * - nz > 0 is the number of decision variables
 * - nv > 0 is the number of inequality constraints
 */
class FBstabDense {
 public:
  FBSTAB_NO_COPY_NO_MOVE_NO_ASSIGN(FBstabDense)
  /** Structure to hold the problem data. */
  struct ProblemData {
    ProblemData(int nz, int nv) {
      H.resize(nz, nz);
      A.resize(nv, nz);
      f.resize(nz);
      b.resize(nv);
    }
    /// nz x nz real positive semidefinite Hessian matrix.
    Eigen::MatrixXd H;
    /// nv x nz real constraint Jacobian.
    Eigen::MatrixXd A;
    /// nz real linear cost.
    Eigen::VectorXd f;
    /// nv real constraint rhs.
    Eigen::VectorXd b;
  };

  /**
   * Structure to hold the initial guess.
   * The vectors will be overwritten with the solution.
   */
  struct Variable {
    Variable(int nz, int nv) {
      z = Eigen::VectorXd::Zero(nz);
      v = Eigen::VectorXd::Zero(nv);
      y = Eigen::VectorXd::Zero(nv);
    }
    /// Decision variables in \reals^nz.
    Eigen::VectorXd z;
    /// Equality duals
    Eigen::VectorXd l;
    /// Inequality duals in \reals^nv.
    Eigen::VectorXd v;
    /// Constraint margin, i.e., y = b-Az, in \reals^nv.
    Eigen::VectorXd y;
  };

  /** A Structure to hold options */
  struct Options : public AlgorithmParameters {};

  /**
   * Allocates needed workspace given the dimensions of the QPs to
   * be solved. Throws a runtime_error if any inputs are non-positive.
   *
   * @param[in] number of decision variables
   * @param[in] number of inequality constraints
   */
  FBstabDense(int nz, int nv);

  /**
   * Solves an instance of (1)
   *
   * @param[in]   qp  problem data
   * @param[in,out] x   initial guess, overwritten with the solution
   * @param[in] use_initial_guess if false the solver is initialized at the
   * origin.
   *
   * @return Summary of the optimizer output, see fbstab_algorithm.h.
   */
  SolverOut Solve(const ProblemData& qp, Variable* x,
                  bool use_initial_guess = true);

  /**
   * Allows for setting of solver options. See fbstab_algorithm.h for
   * a list of adjustable options.
   * @param[in] option New option struct
   */
  void UpdateOptions(const Options& options);

  /** Returns default settings, recommended for most problems. */
  static Options DefaultOptions();
  /** Settings for increased reliability for use on hard problems. */
  static Options ReliableOptions();

 private:
  int nz_ = 0;
  int nv_ = 0;
  int nl_ = 0;

  Options opts_;
  std::unique_ptr<FBstabAlgoDense> algorithm_;
  std::unique_ptr<FullVariable> x1_;
  std::unique_ptr<FullVariable> x2_;
  std::unique_ptr<FullVariable> x3_;
  std::unique_ptr<FullVariable> x4_;
  std::unique_ptr<FullResidual> r1_;
  std::unique_ptr<FullResidual> r2_;
  std::unique_ptr<DenseCholeskySolver> linear_solver_;
  std::unique_ptr<FullFeasibility> feasibility_checker_;
};

}  // namespace fbstab
