#include "fbstab/components/dense_cholesky_solver.h"

#include <Eigen/Dense>
#include <cmath>

#include "fbstab/components/dense_data.h"
#include "fbstab/components/full_residual.h"
#include "fbstab/components/full_variable.h"

namespace fbstab {

DenseCholeskySolver::DenseCholeskySolver(int nz, int nl, int nv)
    : ldlt_(nz + nl) {
  if (nz <= 0 || nv <= 0 || nl < 0) {
    throw std::runtime_error(
        "In DenseCholeskySolver: nz and nv must be > 0 and nl >= 0");
  }
  nz_ = nz;
  nl_ = nl;
  nv_ = nv;

  K_.resize(nz_ + nl_, nz_ + nl_);
  E_.resize(nz_, nz_);
  r1_.resize(nz_ + nl_);
  r2_.resize(nv_);
  Gamma_.resize(nv_);
  mus_.resize(nv_);
  gamma_.resize(nv_);
  B_.resize(nv_, nz_);
}

bool DenseCholeskySolver::Initialize(const FullVariable& x,
                                     const FullVariable& xbar, double sigma) {
  NullDataCheck();
  if (!x.SameSize(xbar)) {
    throw std::runtime_error(
        "In DenseCholeskySolver::Factor: inputs must be the same size");
  }
  if (xbar.nz_ != nz_ || xbar.nv_ != nv_) {
    throw std::runtime_error(
        "In DenseCholeskySolver::Factor: inputs must match object size.");
  }
  if (sigma <= 0) {
    throw std::runtime_error(
        "In DenseCholeskySolver::Factor: sigma must be positive.");
  }
  const auto& H = data_->H_;
  const auto& G = data_->G_;
  const auto& A = data_->A_;

  // E = H + sigma I + A'*diag(Gamma(x))*A
  E_ = H + sigma * Eigen::MatrixXd::Identity(nz_, nz_);
  Eigen::Vector2d pfb_gradient;
  for (int i = 0; i < nv_; i++) {
    const double ys = x.y(i) + sigma * (x.v(i) - xbar.v(i));
    pfb_gradient = PFBGradient(ys, x.v(i));
    gamma_(i) = pfb_gradient(0);
    mus_(i) = pfb_gradient(1) + sigma * pfb_gradient(0);
    Gamma_(i) = gamma_(i) / mus_(i);
  }
  // B is used to avoid temporaries
  B_.noalias() = Gamma_.asDiagonal() * A;
  E_.noalias() += A.transpose() * B_;

  // K = [E G']
  //     [G -S]
  K_.block(0, 0, nz_, nz_) = E_;
  K_.block(nz_, 0, nl_, nz_) = G;
  K_.block(nz_, nz_, nl_, nl_) = -sigma * Eigen::MatrixXd::Identity(nl_, nl_);

  // Factor K = LDL'
  ldlt_.compute(K_);
  Eigen::ComputationInfo status = ldlt_.info();
  if (status != Eigen::Success) {
    return false;
  } else {
    return true;
  }
}

bool DenseCholeskySolver::Solve(const FullResidual& r, FullVariable* x) const {
  if (x == nullptr) {
    throw std::runtime_error(
        "In DenseCholeskySolver::Solve: x cannot be null.");
  }
  if (!r.SameSize(*x)) {
    throw std::runtime_error(
        "In DenseCholeskySolver::Solve residual and variable objects must be "
        "the same size");
  }
  if (x->nz_ != nz_ || x->nv_ != nv_ || x->nl_ != nl_) {
    throw std::runtime_error(
        "In DenseCholeskySolver::Factor: inputs must match object size.");
  }
  const auto& A = data_->A_;
  const auto& b = data_->b_;

  // This method solves the system:
  // [E G'] [z] = [rz - A'*D^-1 * rv]
  // [G -S] [l]   [-rl              ]
  // Dv = rv + C*A*z
  // Where D = diag(mus), C = diag(gamma) and K has been precomputed by the
  // factor routine. See (28) and (29) in https://arxiv.org/pdf/1901.04046.pdf

  // Compute rz - A'*(rv./mus) and store it in r1_.
  r2_ = r.v().cwiseQuotient(mus_);
  // r1_.noalias() = r.z() - A.transpose() * r2_;
  r1_.segment(0, nz_).noalias() = r.z() - A.transpose() * r2_;
  r1_.segment(nz_, nl_) = -r.l();

  // Solve using the precomputed factorization then extract
  r1_ = ldlt_.solve(r1_);
  x->z() = r1_.segment(0, nz_);
  x->l() = r1_.segment(nz_, nl_);

  // Compute v = diag(1/mus) * (rv + diag(gamma)*A*z)
  // written so as to avoid temporary creation
  r2_.noalias() = A * x->z();
  r2_.noalias() = gamma_.asDiagonal() * r2_;
  r2_.noalias() += r.v();
  x->v() = r2_.cwiseQuotient(mus_);

  // y = b - Az
  x->y() = b - A * x->z();

  return true;
}

Eigen::Vector2d DenseCholeskySolver::PFBGradient(double a, double b) const {
  const double r = sqrt(a * a + b * b);
  const double d = 1.0 / sqrt(2.0);

  Eigen::Vector2d v;
  if (r < zero_tolerance_) {
    v(0) = alpha_ * (1.0 - d);
    v(1) = alpha_ * (1.0 - d);

  } else if ((a > 0) && (b > 0)) {
    v(0) = alpha_ * (1.0 - a / r) + (1.0 - alpha_) * b;
    v(1) = alpha_ * (1.0 - b / r) + (1.0 - alpha_) * a;

  } else {
    v(0) = alpha_ * (1.0 - a / r);
    v(1) = alpha_ * (1.0 - b / r);
  }

  return v;
}

void DenseCholeskySolver::NullDataCheck() const {
  if (data_ == nullptr) {
    throw std::runtime_error(
        "DenseCholeskySolver tried to access problem data before it's linked.");
  }
}

}  // namespace fbstab
