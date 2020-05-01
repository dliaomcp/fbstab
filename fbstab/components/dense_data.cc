#include "fbstab/components/dense_data.h"

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace fbstab {

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

DenseData::DenseData(const MatrixXd *H, const VectorXd *f, const MatrixXd *A,
                     const VectorXd *b) {
  if (H == nullptr || f == nullptr || A == nullptr || b == nullptr) {
    throw std::runtime_error("Inputs to DenseData::DenseData cannot be null.");
  }
  if (H->rows() != H->cols() || H->rows() != f->size()) {
    throw std::runtime_error(
        "In DenseData::DenseData: H must be square and the same size as f.");
  }
  if (A->cols() != H->rows() || A->rows() != b->size()) {
    throw std::runtime_error(
        "In DenseData::DenseData: Sizing of data defining Az <= b is "
        "inconsistent.");
  }

  H_ = H;
  A_ = A;
  f_ = f;
  b_ = b;

  nz_ = f->size();
  nv_ = b->size();
  nl_ = 0;

  forcing_norm_ = sqrt(b->squaredNorm() + f->squaredNorm());
}

void DenseData::gemvH(const Eigen::VectorXd &x, double a, double b,
                      Eigen::VectorXd *y) const {
  *y = a * (*H_) * x + b * (*y);
}

void DenseData::gemvA(const Eigen::VectorXd &x, double a, double b,
                      Eigen::VectorXd *y) const {
  *y = a * (*A_) * x + b * (*y);
}

void DenseData::gemvAT(const Eigen::VectorXd &x, double a, double b,
                       Eigen::VectorXd *y) const {
  *y = a * A_->transpose() * x + b * (*y);
}

void DenseData::axpyf(double a, Eigen::VectorXd *y) const { *y += a * (*f_); }

void DenseData::axpyb(double a, Eigen::VectorXd *y) const { *y += a * (*b_); }

// These are no-ops for now.
void DenseData::gemvG(const Eigen::VectorXd &x, double a, double b,
                      Eigen::VectorXd *y) const {
  std::ignore = x;
  std::ignore = a;
  std::ignore = b;
  std::ignore = y;
}

void DenseData::gemvGT(const Eigen::VectorXd &x, double a, double b,
                       Eigen::VectorXd *y) const {
  std::ignore = x;
  std::ignore = a;
  std::ignore = b;
  std::ignore = y;
}

void DenseData::axpyh(double a, Eigen::VectorXd *y) const {
  std::ignore = a;
  std::ignore = y;
}

}  // namespace fbstab
