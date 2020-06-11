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

/**
 * FBstabDense implements the Proximally Stabilized Semismooth Algorithm
 * for solving convex quadratic programs of the following form (1):
 *
 *     min.    1/2  z'Hz + f'z
 *     s.t.    Gz = h
 *             Az <= b
 *
 * where H is symmetric and positive semidefinite and its dual
 *
 *     min.   1/2  z'Hz + b'v + h'l
 *     s.t.   Hz + f + A'v + G'l = 0
 *            v >= 0.
 *
 * Or equivalently for solving its KKT system
 *
 *     Hz + f + A'v + G'l = 0
 *     h - Gz = 0
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
 * The problem is of size (nz, nl, nv) where:
 * - nz > 0 is the number of decision variables
 * - nl >= 0 is the number of equality constraints
 * - nv > 0 is the number of inequality constraints
 */
class FBstabDense {
 public:
  FBSTAB_NO_COPY_NO_MOVE_NO_ASSIGN(FBstabDense)

  /** Structure to hold the problem data. */
  struct ProblemData {
    ProblemData() = default;
    ProblemData(int nz, int nl, int nv);
    Eigen::MatrixXd H;  /// nz x nz positive semidefinite Hessian matrix.
    Eigen::MatrixXd G;  /// nl x nz equality Jacobian
    Eigen::MatrixXd A;  /// nv x nz inequality Jacobian.
    Eigen::VectorXd f;  /// nz linear cost.
    Eigen::VectorXd h;  /// nl equality rhs
    Eigen::VectorXd b;  /// nv inequality rhs.
  };

  /** Problem data using preallocated memory */
  struct ProblemDataRef {
    ProblemDataRef() = delete;
    ProblemDataRef(const Eigen::Map<Eigen::MatrixXd>* H_,
                   const Eigen::Map<Eigen::VectorXd>* f_,
                   const Eigen::Map<Eigen::MatrixXd>* G_,
                   const Eigen::Map<Eigen::VectorXd>* h_,
                   const Eigen::Map<Eigen::MatrixXd>* A_,
                   const Eigen::Map<Eigen::VectorXd>* b_);

    Eigen::Map<const Eigen::MatrixXd> H;  /// nz x nz Hessian
    Eigen::Map<const Eigen::MatrixXd> G;  /// nl x nz equality Jacobian
    Eigen::Map<const Eigen::MatrixXd> A;  /// nv x nz inequality Jacobian.
    Eigen::Map<const Eigen::VectorXd> f;  /// nz linear cost.
    Eigen::Map<const Eigen::VectorXd> h;  /// nl equality rhs
    Eigen::Map<const Eigen::VectorXd> b;  /// nv inequality rhs.
  };

  /** Structure to hold the initial guess. */
  struct Variable {
    /** Allocates a variable of size (nz,nl,nv) at the origin.  */
    Variable(int nz, int nl, int nv);
    Eigen::VectorXd z;  /// Decision variables in \reals^nz.
    Eigen::VectorXd l;  /// Equality duals \in reals^nl
    Eigen::VectorXd v;  /// Inequality duals in \reals^nv.
    Eigen::VectorXd y;  /// Constraint margin, i.e., y = b-Az, in \reals^nv.
  };

  /** An input structure for reusing preallocated memory. */
  struct VariableRef {
    VariableRef() = delete;
    VariableRef(Eigen::Map<Eigen::VectorXd>* z_,
                Eigen::Map<Eigen::VectorXd>* l_,
                Eigen::Map<Eigen::VectorXd>* v_,
                Eigen::Map<Eigen::VectorXd>* y_);
    // Fill all fields with a.
    void fill(double a);
    Eigen::Map<Eigen::VectorXd> z;  /// Decision variables in \reals^nz.
    Eigen::Map<Eigen::VectorXd> l;  /// Equality duals
    Eigen::Map<Eigen::VectorXd> v;  /// Inequality duals in \reals^nv.
    Eigen::Map<Eigen::VectorXd> y;  /// y = b-Az, in \reals^nv.
  };

  /**
   * A Structure to hold options. See fbstab_algorithm.h for more details.
   */
  struct Options : public AlgorithmParameters {};

  /**
   * Allocates needed workspace given the dimensions of the QPs to
   * be solved. Throws a runtime_error if any inputs are non-positive.
   *
   * @param[in] nz number of decision variables
   * @param[in] nl number of equality constraints
   * @param[in] nv number of inequality constraints
   */
  FBstabDense(int nz, int nl, int nv);

  /**
   * Solves an instance of (1)
   *
   * @param[in]   qp  problem data
   * @param[in,out] x   initial guess, overwritten with the solution
   *
   * @return Summary of the optimizer output, see fbstab_algorithm.h.
   *
   * The template parameter allows for both Variable and VariableRef type
   * inputs.
   */
  template <class InputData, class InputVariable>
  SolverOut Solve(const InputData& qp, InputVariable* x) {
    // Data performs its own validation checks.
    DenseData data(&qp.H, &qp.f, &qp.G, &qp.h, &qp.A, &qp.b);
    ValidateInputs(data, *x);
    return algorithm_->Solve(data, &x->z, &x->l, &x->v, &x->y);
  }

  /**
   * Allows for setting of solver options. See fbstab_algorithm.h for
   * a list of adjustable options.
   *
   * @param[in] option New options struct
   */
  void UpdateOptions(const Options& options);

  /** Returns default settings, recommended for most problems. */
  static Options DefaultOptions();
  /** Settings for increased reliability for use on hard problems. */
  static Options ReliableOptions();

 private:
  int nz_ = 0;
  int nl_ = 0;
  int nv_ = 0;

  template <class InputVariable>
  void ValidateInputs(const DenseData& data, const InputVariable& x) {
    if (nz_ != data.nz() || nv_ != data.nv() || nl_ != data.nl()) {
      throw std::runtime_error(
          "In FBstabDense::Solve: mismatch between *this and data dimensions.");
    }
    if (nz_ != x.z.size() || x.l.size() != nl_ || nv_ != x.v.size()) {
      throw std::runtime_error(
          "In FBstabDense::Solve: mismatch between *this and initial guess "
          "dimensions.");
    }
  }

  using FBstabAlgoDense = FBstabAlgorithm<FullVariable, FullResidual,
                                          DenseCholeskySolver, FullFeasibility>;
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
