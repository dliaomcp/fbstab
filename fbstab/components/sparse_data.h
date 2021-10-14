#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "fbstab/components/abstract_components.h"
#include "tools/copyable_macros.h"

namespace fbstab {

class SparseData : public Data {
 public:
  template <class Matrix, class Vector>
  SparseData(const Matrix* H, const Vector* f, const Matrix* G, const Vector* h,
             const Matrix* A, const Vector* b);

  /** Performs the operation y <- a*H*x + b*y */
  void gemvH(const Eigen::VectorXd& x, double a, double b,
             Eigen::VectorXd* y) const;

  /** Performs the operation y <- a*A*x + b*y */
  void gemvA(const Eigen::VectorXd& x, double a, double b,
             Eigen::VectorXd* y) const;

  /** Performs the operation y <- a*A'*x + b*y */
  void gemvAT(const Eigen::VectorXd& x, double a, double b,
              Eigen::VectorXd* y) const;

  /** Performs the operation y <- a*G*x + b*y */
  void gemvG(const Eigen::VectorXd& x, double a, double b,
             Eigen::VectorXd* y) const;

  /** Performs the operation y <- a*G'*x + b*y */
  void gemvGT(const Eigen::VectorXd& x, double a, double b,
              Eigen::VectorXd* y) const;

  /** Performs the operation y <- a*f + y */
  void axpyf(double a, Eigen::VectorXd* y) const;

  /** Performs the operation y <- a*b + y */
  void axpyb(double a, Eigen::VectorXd* y) const;

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

  const Eigen::Map<const Eigen::SparseMatrix<double>> H_;
  const Eigen::Map<const Eigen::SparseMatrix<double>> H_;
  const Eigen::Map<const Eigen::SparseMatrix<double>> H_;

  const Eigen::Map<const Eigen::VectorXd> f_;
  const Eigen::Map<const Eigen::VectorXd> f_;
  const Eigen::Map<const Eigen::VectorXd> f_;
};

}  // namespace fbstab