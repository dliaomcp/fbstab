#include "fbstab/components/full_variable.h"

#include <Eigen/Dense>
#include <cmath>
#include <memory>

#include "tools/utilities.h"

namespace fbstab {

using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;

FullVariable::FullVariable(int nz, int nl, int nv) {
  if (nz <= 0 || nl < 0 || nv <= 0) {
    throw std::runtime_error(
        "All size inputs to FullVariable::FullVariable must be >= 1.");
  }

  nz_ = nz;
  nl_ = nl;
  nv_ = nv;

  z_storage_ = tools::make_unique<VectorXd>(nz_);
  l_storage_ = tools::make_unique<VectorXd>(nl_);
  v_storage_ = tools::make_unique<VectorXd>(nv_);
  y_storage_ = tools::make_unique<VectorXd>(nv_);

  z_ = z_storage_.get();
  l_ = l_storage_.get();
  v_ = v_storage_.get();
  y_ = y_storage_.get();

  z_->setConstant(0.0);
  l_->setConstant(0.0);
  v_->setConstant(0.0);
  y_->setConstant(0.0);
}

FullVariable::FullVariable(VectorXd* z, VectorXd* l, VectorXd* v, VectorXd* y) {
  if (z == nullptr || l == nullptr || v == nullptr || y == nullptr) {
    throw std::runtime_error(
        "Inputs to FullVariable::FullVariable cannot be null.");
  }
  if (z->size() == 0 || v->size() == 0 || y->size() == 0) {
    throw std::runtime_error("Invalid input to FullVariable.");
  }
  if (v->size() != y->size()) {
    throw std::runtime_error(
        "In FullVariable::FullVariable, y and v must be the same size");
  }

  nz_ = z->size();
  nl_ = l->size();
  nv_ = v->size();

  z_ = z;
  l_ = l;
  v_ = v;
  y_ = y;
}

void FullVariable::Fill(double a) {
  z_->setConstant(a);
  l_->setConstant(a);
  v_->setConstant(a);
  InitializeConstraintMargin();
}

void FullVariable::InitializeConstraintMargin() {
  NullDataCheck();
  // y = b - A*z
  y_->setConstant(0.0);
  data_->axpyb(1.0, y_);
  data_->gemvA(*z_, -1.0, 1.0, y_);
}

void FullVariable::axpy(double a, const FullVariable& x) {
  NullDataCheck();

  z_->noalias() += a * (*x.z_);
  l_->noalias() += a * (*x.l_);
  v_->noalias() += a * (*x.v_);

  // y <- y + a*(x.y - b)
  y_->noalias() += a * (*x.y_);
  data_->axpyb(-a, y_);
}

void FullVariable::Copy(const FullVariable& x) {
  *z_ = *x.z_;
  *l_ = *x.l_;
  *v_ = *x.v_;
  *y_ = *x.y_;
  data_ = x.data_;
}

void FullVariable::ProjectDuals() { *v_ = v_->cwiseMax(0); }

double FullVariable::Norm() const {
  const double t1 = z_->norm();
  const double t2 = l_->norm();
  const double t3 = v_->norm();

  return sqrt(t1 * t1 + t2 * t2 + t3 * t3);
}

bool FullVariable::SameSize(const FullVariable& x) const {
  return (x.nz_ == nz_ && x.nl_ == nl_ && x.nv_ == nv_);
}

void FullVariable::NullDataCheck() const {
  if (data_ == nullptr) {
    throw std::runtime_error(
        "FullVariable tried to access problem data before it's "
        "linked.");
  }
}

}  // namespace fbstab
