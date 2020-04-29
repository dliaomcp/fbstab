#include "fbstab/components/full_feasibility.h"

#include <Eigen/Dense>
#include <algorithm>
#include <stdexcept>

#include "fbstab/components/abstract_components.h"
#include "fbstab/components/full_variable.h"

namespace fbstab {

FullFeasibility::FullFeasibility(int nz, int nl, int nv) {
  if (nz <= 0 || nl < 0 || nv <= 0) {
    throw std::runtime_error("Incorrect size inputs to FullFeasibility.");
  }
  nz_ = nz;
  nl_ = nl;
  nv_ = nv;

  tz_.resize(nz_);
  tl_.resize(nl_);
  tv_.resize(nv_);
}

FullFeasibility::FeasibilityStatus FullFeasibility::CheckFeasibility(
    const FullVariable& x, double tol) {
  NullDataCheck();

  // The conditions for dual-infeasibility are:
  // max(Az) <= 0 and f'*z < 0 and |Hz| <= tol * |z| and |Gz| <= tol*|z|

  // Compute d1 = max(Az).
  data_->gemvA(x.z(), 1.0, 0.0, &tv_);
  const double d1 = tv_.maxCoeff();

  // Compute d2 = infnorm(Gz).
  data_->gemvG(x.z(), 1.0, 0.0, &tl_);
  const double d2 = tl_.lpNorm<Eigen::Infinity>();

  // Compute d3 = infnorm(Hz).
  data_->gemvH(x.z(), 1.0, 0.0, &tz_);
  const double d3 = tz_.lpNorm<Eigen::Infinity>();

  // Compute d4 = f'*z
  tz_.setConstant(0.0);
  data_->axpyf(1.0, &tz_);
  const double d4 = tz_.dot(x.z());

  double w = x.z().lpNorm<Eigen::Infinity>();
  bool dual_feasible = true;
  if ((d1 <= w * tol) && (d2 <= tol * w) && (d3 <= tol * w) && (d4 < 0) &&
      (w > 1e-14)) {
    dual_feasible = false;
  }

  // The conditions for primal infeasibility are:
  // v'*b + l'*h < 0 and |A'*v + G'*l| \leq tol * |(v,l)|

  // Compute p1 = infnorm(G'*l + A'*v).
  tz_.fill(0.0);
  data_->gemvAT(x.v(), 1.0, 1.0, &tz_);
  data_->gemvGT(x.l(), 1.0, 1.0, &tz_);
  const double p1 = tz_.lpNorm<Eigen::Infinity>();

  // Compute p2 = v'*b + l'*h.
  tv_.setConstant(0.0);
  data_->axpyb(1.0, &tv_);
  tl_.setConstant(0.0);
  data_->axpyh(1.0, &tl_);
  const double p2 = tl_.dot(x.l()) + tv_.dot(x.v());

  const double u = std::max(x.v().lpNorm<Eigen::Infinity>(),
                            x.l().lpNorm<Eigen::Infinity>());
  bool primal_feasible = true;
  if ((p1 <= tol * u) && (p2 < 0)) {
    primal_feasible = false;
  }

  if (primal_feasible && dual_feasible) {
    return FeasibilityStatus::FEASIBLE;
  } else if (primal_feasible && !dual_feasible) {
    return FeasibilityStatus::DUAL_INFEASIBLE;
  } else if (!primal_feasible && dual_feasible) {
    return FeasibilityStatus::PRIMAL_INFEASIBLE;
  } else {
    return FeasibilityStatus::BOTH;
  }
}

void FullFeasibility::NullDataCheck() const {
  if (data_ == nullptr) {
    throw std::runtime_error(
        "FullFeasibility tried to access problem data before it's "
        "linked.");
  }
}

}  // namespace fbstab
