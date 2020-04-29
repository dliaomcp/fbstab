#include "fbstab/components/full_residual.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "fbstab/components/full_variable.h"

namespace fbstab {

FullResidual::FullResidual(int nz, int nl, int nv) {
  if (nz <= 0 || nl < 0 || nv <= 0) {
    throw std::runtime_error(
        "All inputs to FullResidual::FullResidual must be >= 1.");
  }
  nz_ = nz;
  nl_ = nl;
  nv_ = nv;

  z_.resize(nz_);
  l_.resize(nl_);
  v_.resize(nv_);

  Fill(0.0);
}

void FullResidual::Fill(double a) {
  z_.setConstant(a);
  l_.setConstant(a);
  v_.setConstant(a);
}

void FullResidual::Negate() {
  z_ *= -1;
  l_ *= -1;
  v_ *= -1;
}

double FullResidual::Norm() const {
  return sqrt(znorm_ * znorm_ + lnorm_ * lnorm_ + vnorm_ * vnorm_);
}

double FullResidual::Merit() const {
  const double temp = this->Norm();
  return 0.5 * temp * temp;
}

void FullResidual::InnerResidual(const FullVariable& x,
                                 const FullVariable& xbar, double sigma) {
  NullDataCheck();
  // r.z = H*z + f + G'*l + A'*v + sigma*(z-zbar)
  z_.setConstant(0.0);
  data_->axpyf(1.0, &z_);
  data_->gemvH(x.z(), 1.0, 1.0, &z_);
  data_->gemvGT(x.l(), 1.0, 1.0, &z_);
  data_->gemvAT(x.v(), 1.0, 1.0, &z_);
  z_.noalias() += sigma * (x.z() - xbar.z());

  // r.l = h - G*z + sigma(l - lbar)
  l_.setConstant(0.0);
  data_->axpyh(1.0, &l_);
  data_->gemvG(x.z(), -1.0, 1.0, &l_);
  l_.noalias() += sigma * (x.l() - xbar.l());

  // rv = phi(y + sigma*(v-vbar),v)
  for (int i = 0; i < nv_; i++) {
    const double ys = x.y()(i) + sigma * (x.v()(i) - xbar.v()(i));
    v_(i) = pfb(ys, x.v()(i), alpha_);
  }
  znorm_ = z_.norm();
  lnorm_ = l_.norm();
  vnorm_ = v_.norm();
}

void FullResidual::NaturalResidual(const FullVariable& x) {
  NullDataCheck();
  // r.z = H*z + f + G'*l + A'*v
  z_.setConstant(0.0);
  data_->axpyf(1.0, &z_);
  data_->gemvH(x.z(), 1.0, 1.0, &z_);
  data_->gemvGT(x.l(), 1.0, 1.0, &z_);
  data_->gemvAT(x.v(), 1.0, 1.0, &z_);

  // r.l = h - G*z + sigma(l - lbar)
  l_.setConstant(0.0);
  data_->axpyh(1.0, &l_);
  data_->gemvG(x.z(), -1.0, 1.0, &l_);

  // rv = min(y,v)
  for (int i = 0; i < nv_; i++) {
    v_(i) = std::min(x.y()(i), x.v()(i));
  }
  znorm_ = z_.norm();
  lnorm_ = l_.norm();
  vnorm_ = v_.norm();
}

void FullResidual::PenalizedNaturalResidual(const FullVariable& x) {
  NaturalResidual(x);

  for (int i = 0; i < nv_; i++) {
    v_(i) = alpha_ * v_(i) +
            (1 - alpha_) * std::max(0.0, x.y()(i)) * std::max(0.0, x.v()(i));
  }
  znorm_ = z_.norm();
  lnorm_ = l_.norm();
  vnorm_ = v_.norm();
}

double FullResidual::pfb(double a, double b, double alpha) {
  double fb = a + b - sqrt(a * a + b * b);
  return alpha * fb + (1.0 - alpha) * std::max(0.0, a) * std::max(0.0, b);
}

void FullResidual::NullDataCheck() const {
  if (data_ == nullptr) {
    throw std::runtime_error(
        "FullResidual tried to access problem data before it's "
        "linked.");
  }
}

}  // namespace fbstab
