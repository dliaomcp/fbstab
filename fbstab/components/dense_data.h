#pragma once

#include <Eigen/Dense>

#include "fbstab/components/abstract_components.h"
#include "tools/copyable_macros.h"

namespace fbstab {

/**
 * Represents data for quadratic programing problems of the following type (1):
 *
 * min.    1/2  z'Hz + f'z
 * s.t.         Az <= b
 *
 * where H is symmetric and positive semidefinite.
 */

class DenseData : public Data {
 public:
  FBSTAB_NO_COPY_NO_MOVE_NO_ASSIGN(DenseData)
  /**
   * Stores the problem data and performs input validation.
   * This class assumes that the pointers to the data remain valid.
   *
   * @param[in] H Hessian matrix
   * @param[in] f Linear term
   * @param[in] A Constraint matrix
   * @param[in] b Constraint vector
   *
   * Throws a runtime exception if any of the inputs are null or if
   * the sizes of the inputs are inconsistent.
   */
  DenseData(const Eigen::MatrixXd* H, const Eigen::VectorXd* f,
            const Eigen::MatrixXd* A, const Eigen::VectorXd* b);

  /** Performs the operation y <- a*H*x + b*y */
  void gemvH(const Eigen::VectorXd& x, double a, double b,
             Eigen::VectorXd* y) const;

  /** Performs the operation y <- a*A*x + b*y */
  void gemvA(const Eigen::VectorXd& x, double a, double b,
             Eigen::VectorXd* y) const;

  /** Performs the operation y <- a*A'*x + b*y */
  void gemvAT(const Eigen::VectorXd& x, double a, double b,
              Eigen::VectorXd* y) const;

  /** Performs the operation y <- a*f + y */
  void axpyf(double a, Eigen::VectorXd* y) const;

  /** Performs the operation y <- a*b + y */
  void axpyb(double a, Eigen::VectorXd* y) const;

  // These are no-ops for the time being
  /** Performs the operation y <- a*G*x + b*y */
  void gemvG(const Eigen::VectorXd& x, double a, double b,
             Eigen::VectorXd* y) const;

  /** Performs the operation y <- a*G'*x + b*y */
  void gemvGT(const Eigen::VectorXd& x, double a, double b,
              Eigen::VectorXd* y) const;

  /** Performs the operation y <- a*h + y */
  void axpyh(double a, Eigen::VectorXd* y) const;

  double ForcingNorm() const { return forcing_norm_; }

  int nz() const { return nz_; }
  int nl() const { return nl_; }
  int nv() const { return nv_; }

 private:
  int nz_ = 0;  // Number of decision variables.
  int nl_ = 0;  // Number of equality constraints
  int nv_ = 0;  // Number of inequality constraints.

  double forcing_norm_ = 0.0;

  const Eigen::MatrixXd* H_ = nullptr;
  const Eigen::VectorXd* f_ = nullptr;
  const Eigen::MatrixXd* A_ = nullptr;
  const Eigen::VectorXd* b_ = nullptr;

  friend class DenseCholeskySolver;
};

}  // namespace fbstab
