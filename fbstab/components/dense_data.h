#pragma once

#include <Eigen/Dense>

#include "fbstab/components/abstract_components.h"
#include "tools/copyable_macros.h"

namespace fbstab {

/**
 * Represents data for quadratic programing problems of the following type (1):
 *
 * min.    1/2  z'Hz + f'z
 * s.t.    Gz = h
 *         Az <= b
 *
 * where H is symmetric and positive semidefinite.
 */
class DenseData : public Data {
 public:
  FBSTAB_NO_COPY_NO_MOVE_NO_ASSIGN(DenseData)
  DenseData() = delete;

  /**
   * Stores the problem data and performs input validation.
   * This class assumes that the pointers to the data remain valid.
   *
   * @param[in] H Hessian matrix
   * @param[in] f Linear term
   * @param[in] G Constraint matrix
   * @param[in] h Constraint vector
   * @param[in] A Constraint matrix
   * @param[in] b Constraint vector
   *
   * @tparam Matrix: Used to allow MatrixXd or Map<MatrixXd>
   * @tparam Vector: Used to allow VectorXd or Map<VectorXd>
   *
   * The template parameters are explicitly instatiated at the end of the .cc
   * file for MatrixXd/VectorXd and their mapped versions.
   *
   * Throws a runtime exception if any of the inputs are null or if the
   * sizes of the inputs are inconsistent.
   */
  template <class Matrix, class Vector>
  DenseData(const Matrix* H, const Vector* f, const Matrix* G, const Vector* h,
            const Matrix* A, const Vector* b)
      : H_(H->data(), H->rows(), H->cols()),
        G_(G->data(), G->rows(), G->cols()),
        A_(A->data(), A->rows(), A->cols()),
        f_(f->data(), f->size()),
        h_(h->data(), h->size()),
        b_(b->data(), b->size()) {
    if (H->rows() != H->cols() || H->rows() != f->size()) {
      throw std::runtime_error(
          "In DenseData::DenseData: H must be square and the same size as f");
    }
    if (A->cols() != H->rows() || A->rows() != b->size()) {
      throw std::runtime_error(
          "In DenseData::DenseData: Sizing of data defining Az <= b is "
          "inconsistent.");
    }
    if (G->cols() != H->rows() || G->rows() != h->size()) {
      throw std::runtime_error(
          "In DenseData::DenseData: Sizing of Gz = h is "
          "inconsistent.");
    }

    nz_ = f->size();
    nl_ = h->size();
    nv_ = b->size();

    forcing_norm_ =
        sqrt(b->squaredNorm() + f->squaredNorm() + h->squaredNorm());
  }

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

  const Eigen::Map<const Eigen::MatrixXd> H_;
  const Eigen::Map<const Eigen::MatrixXd> G_;
  const Eigen::Map<const Eigen::MatrixXd> A_;
  const Eigen::Map<const Eigen::VectorXd> f_;
  const Eigen::Map<const Eigen::VectorXd> h_;
  const Eigen::Map<const Eigen::VectorXd> b_;

  friend class DenseCholeskySolver;
};

}  // namespace fbstab
