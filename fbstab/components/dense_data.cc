#include "fbstab/components/dense_data.h"

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

namespace fbstab {

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

void DenseData::gemvH(const Eigen::VectorXd &x, double a, double b,
                      Eigen::VectorXd *y) const {
  *y = a * H_ * x + b * (*y);
}

void DenseData::gemvG(const Eigen::VectorXd &x, double a, double b,
                      Eigen::VectorXd *y) const {
  *y = a * G_ * x + b * (*y);
}

void DenseData::gemvGT(const Eigen::VectorXd &x, double a, double b,
                       Eigen::VectorXd *y) const {
  *y = a * G_.transpose() * x + b * (*y);
}

void DenseData::gemvA(const Eigen::VectorXd &x, double a, double b,
                      Eigen::VectorXd *y) const {
  *y = a * A_ * x + b * (*y);
}

void DenseData::gemvAT(const Eigen::VectorXd &x, double a, double b,
                       Eigen::VectorXd *y) const {
  *y = a * A_.transpose() * x + b * (*y);
}

void DenseData::axpyf(double a, Eigen::VectorXd *y) const { *y += a * f_; }

void DenseData::axpyh(double a, Eigen::VectorXd *y) const { *y += a * h_; }

void DenseData::axpyb(double a, Eigen::VectorXd *y) const { *y += a * b_; }

}  // namespace fbstab
