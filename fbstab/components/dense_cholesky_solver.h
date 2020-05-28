#pragma once

#include <Eigen/Dense>

#include "fbstab/components/abstract_components.h"
#include "fbstab/components/dense_data.h"
#include "fbstab/components/full_residual.h"
#include "fbstab/components/full_variable.h"
#include "tools/copyable_macros.h"

namespace fbstab {

// Forward declaration of testing class to enable a friend declaration.
namespace test {
class DenseComponentUnitTests;
}  // namespace test

/**
 * A class for computing the search directions used by the FBstab QP Solver.
 * It solves the systems of linear equations described in (28) and (29) of
 * https://arxiv.org/pdf/1901.04046.pdf.
 *
 * This class allocates its own workspace memory and splits step computation
 * into solve and factor steps to allow for solving with multiple
 * right hand sides.
 *
 * This class has mutable fields and is thus not thread safe.
 */
class DenseCholeskySolver
    : public LinearSolver<FullVariable, FullResidual, DenseData> {
 public:
  FBSTAB_NO_COPY_NO_MOVE_NO_ASSIGN(DenseCholeskySolver)
  /**
   * Allocates workspace memory.
   * @param [nz] Number of decision variables > 0
   * @param [nl] Number of equality constraints >= 0
   * @param [nv] Number of inequality constraints > 0
   */
  DenseCholeskySolver(int nz, int nl, int nv);

  /**
   * Links to problem data needed to perform calculations.
   * Calculations cannot be performed until a data object is provided.
   * @param[in] data pointer to the problem data
   */
  void LinkData(const DenseData* data) { data_ = data; }

  /**
   * Factors the matrix V(x,xbar,sigma) using a Schur complement approach
   * followed by a Cholesky factorization and stores the factorization
   * internally.
   *
   * The matrix V is computed as described in
   * Algorithm 4 of https://arxiv.org/pdf/1901.04046.pdf.
   *
   * @param[in]  x       Inner loop iterate
   * @param[in]  xbar    Outer loop iterate
   * @param[in]  sigma   Regularization strength
   * @return             true if factorization succeeds false otherwise.
   *
   * Throws a runtime_error if x and xbar aren't the correct size,
   * sigma is negative or the problem data isn't linked.
   */
  bool Initialize(const FullVariable& x, const FullVariable& xbar,
                  double sigma);

  /**
   * Solves the system V*x = r and stores the result in x.
   * This method assumes that the Factor routine was run to
   * compute then factor the matrix V.
   *
   * @param[in]   r   The right hand side vector
   * @param[out]  x   Overwritten with the solution
   * @return true if successful, false otherwise
   *
   * Throws a runtime_error if x and r aren't the correct sizes,
   * if x is null or if the problem data isn't linked.
   */
  bool Solve(const FullResidual& r, FullVariable* x) const;

  /**
   * Sets the alpha parameter defined in (19)
   * of https://arxiv.org/pdf/1901.04046.pdf.
   */
  void SetAlpha(double alpha) { alpha_ = alpha; }

 private:
  friend class test::DenseComponentUnitTests;
  int nz_ = 0;  // number of decision variables
  int nl_ = 0;  // number of equality constraints
  int nv_ = 0;  // number of inequality constraints

  double alpha_ = 0.95;
  const double zero_tolerance_ = 1e-13;
  const DenseData* data_ = nullptr;

  // workspace variables
  Eigen::MatrixXd K_;
  Eigen::MatrixXd E_;
  Eigen::LDLT<Eigen::MatrixXd> ldlt_;
  mutable Eigen::VectorXd r1_;
  mutable Eigen::VectorXd r2_;
  Eigen::VectorXd Gamma_;
  Eigen::VectorXd mus_;
  Eigen::VectorXd gamma_;
  Eigen::MatrixXd B_;

  // Computes the gradient of the penalized fischer-burmeister (PFB)
  // function, (19) in https://arxiv.org/pdf/1901.04046.pdf.
  // See section 3.3.
  Eigen::Vector2d PFBGradient(double a, double b) const;

  void NullDataCheck() const;
};

}  // namespace fbstab
