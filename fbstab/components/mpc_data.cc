#include "fbstab/components/mpc_data.h"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "tools/matrix_sequence.h"

namespace fbstab {

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;
using Map = Eigen::Map<Eigen::MatrixXd>;
using ConstMap = Eigen::Map<const Eigen::MatrixXd>;

void MpcData::gemvH(const Eigen::VectorXd& x, double a, double b,
                    Eigen::VectorXd* y) const {
  if (y == nullptr) {
    throw std::runtime_error("In MpcData::gemvH: y input is null.");
  }
  if (x.size() != nz_ || y->size() != nz_) {
    throw std::runtime_error("Size mismatch in MpcData::gemvH.");
  }
  if (b == 0.0) {
    y->fill(0.0);
  } else if (b != 1.0) {
    (*y) *= b;
  }

  // Create reshaped views of input and output vectors.
  Map w(y->data(), nx_ + nu_,
        N_ + 1);  // w = reshape(y, [nx + nu, N + 1]);
  ConstMap v(x.data(), nx_ + nu_,
             N_ + 1);  // v = reshape(x, [nx + nu, N + 1]);
  for (int i = 0; i < N_ + 1; i++) {
    const auto& Q = Q_(i);
    const auto& S = S_(i);
    const auto& R = R_(i);

    // These variables alias w.
    auto yx = w.block(0, i, nx_, 1);
    auto yu = w.block(nx_, i, nu_, 1);

    // These variables alias v.
    const auto vx = v.block(0, i, nx_, 1);
    const auto vu = v.block(nx_, i, nu_, 1);

    // [yx] += a * [Q(i) S(i)'] [vx]
    // [yu]        [S(i) R(i) ] [vu]
    // Using lazyProduct is inefficient so should be avoided when possible.
    if (a == 1.0) {
      yx.noalias() += Q * vx + S.transpose() * vu;
      yu.noalias() += S * vx + R * vu;
    } else if (a == -1.0) {
      yx.noalias() -= Q * vx + S.transpose() * vu;
      yu.noalias() -= S * vx + R * vu;
    } else {
      yx += a * Q.lazyProduct(vx);
      yx += a * S.transpose().lazyProduct(vu);
      yu += a * S.lazyProduct(vx);
      yu += a * R.lazyProduct(vu);
    }
  }
}

void MpcData::gemvA(const Eigen::VectorXd& x, double a, double b,
                    Eigen::VectorXd* y) const {
  if (y == nullptr) {
    throw std::runtime_error("In MpcData::gemvA: y input is null.");
  }
  if (x.size() != nz_ || y->size() != nv_) {
    throw std::runtime_error("Size mismatch in MpcData::gemvA.");
  }
  if (b == 0.0) {
    y->fill(0.0);
  } else if (b != 1.0) {
    (*y) *= b;
  }
  // Create reshaped views of input and output vectors.
  ConstMap z(x.data(), nx_ + nu_, N_ + 1);
  Map w(y->data(), nc_, N_ + 1);

  for (int i = 0; i < N_ + 1; i++) {
    const auto& E = E_(i);
    const auto& L = L_(i);

    // This aliases w.
    auto yi = w.col(i);

    // These alias z.
    const auto xi = z.block(0, i, nx_, 1);
    const auto ui = z.block(nx_, i, nu_, 1);

    // yi += a*(E*vx + L*vu)
    if (a == 1.0) {
      yi.noalias() += E * xi + L * ui;
    } else if (a == -1.0) {
      yi.noalias() -= E * xi + L * ui;
    } else {
      yi += a * E.lazyProduct(xi);
      yi += a * L.lazyProduct(ui);
    }
  }
}

void MpcData::gemvG(const Eigen::VectorXd& x, double a, double b,
                    Eigen::VectorXd* y) const {
  if (y == nullptr) {
    throw std::runtime_error("In MpcData::gemvG: y input is null.");
  }
  if (x.size() != nz_ || y->size() != nl_) {
    throw std::runtime_error("Size mismatch in MpcData::gemvG.");
  }
  if (b == 0.0) {
    y->fill(0.0);
  } else if (b != 1.0) {
    (*y) *= b;
  }
  // Create reshaped views of input and output vectors.
  ConstMap z(x.data(), nx_ + nu_, N_ + 1);
  Map w(y->data(), nx_, N_ + 1);

  w.col(0).noalias() += -a * z.block(0, 0, nx_, 1);

  for (int i = 1; i < N_ + 1; i++) {
    const auto& A = A_(i - 1);
    const auto& B = B_(i - 1);

    // Alias for the output at stage i.
    auto yi = w.col(i);
    // Aliases for the state and control at stage i - 1.
    const auto xm1 = z.block(0, i - 1, nx_, 1);
    const auto um1 = z.block(nx_, i - 1, nu_, 1);
    // Alias for the state at stage i.
    const auto xi = z.block(0, i, nx_, 1);

    // y(i) += a*(A(i-1)*x(i-1) + B(i-1)u(i-1) - x(i))
    if (a == 1.0) {
      yi.noalias() += A * xm1 + B * um1;
      yi.noalias() -= xi;
    } else if (a == -1.0) {
      yi.noalias() -= A * xm1 + B * um1;
      yi.noalias() += xi;
    } else {
      yi += a * A.lazyProduct(xm1);
      yi += a * B.lazyProduct(um1);
      yi -= a * xi;
    }
  }
}

void MpcData::gemvGT(const Eigen::VectorXd& x, double a, double b,
                     Eigen::VectorXd* y) const {
  if (y == nullptr) {
    throw std::runtime_error("In MpcData::gemvGT: y input is null.");
  }
  if (x.size() != nl_ || y->size() != nz_) {
    throw std::runtime_error("Size mismatch in MpcData::gemvGT.");
  }
  if (b == 0.0) {
    y->fill(0.0);
  } else if (b != 1.0) {
    (*y) *= b;
  }

  // Create reshaped views of input and output vectors.
  ConstMap v(x.data(), nx_, N_ + 1);
  Map w(y->data(), nx_ + nu_, N_ + 1);

  for (int i = 0; i < N_; i++) {
    const auto& A = A_(i);
    const auto& B = B_(i);

    // Aliases for the dual variables at stage i and i+1;
    const auto vi = v.col(i);
    const auto vp1 = v.col(i + 1);

    // Aliases for the state and control at stage i.
    auto xi = w.block(0, i, nx_, 1);
    auto ui = w.block(nx_, i, nu_, 1);

    // x(i) += a*(-v(i) + A(i)' * v(i+1))
    // u(i) += a*B(i)' * v(i+1)
    xi.noalias() += -a * vi;
    if (a == 1.0) {
      xi.noalias() += A.transpose() * vp1;
      ui.noalias() += B.transpose() * vp1;
    } else if (a == -1.0) {
      xi.noalias() -= A.transpose() * vp1;
      ui.noalias() -= B.transpose() * vp1;
    } else {
      xi.noalias() += a * A.transpose().lazyProduct(vp1);
    }
  }
  // The i = N step of the recursion.
  w.block(0, N_, nx_, 1).noalias() += -a * v.col(N_);
}

void MpcData::gemvAT(const Eigen::VectorXd& x, double a, double b,
                     Eigen::VectorXd* y) const {
  if (y == nullptr) {
    throw std::runtime_error("In MpcData::gemvAT: y input is null.");
  }
  if (x.size() != nv_ || y->size() != nz_) {
    throw std::runtime_error("Size mismatch in MpcData::gemvAT.");
  }
  if (b == 0.0) {
    y->fill(0.0);
  } else if (b != 1.0) {
    (*y) *= b;
  }
  // Create reshaped views of input and output vectors.
  ConstMap v(x.data(), nc_, N_ + 1);
  Map w(y->data(), nx_ + nu_, N_ + 1);

  for (int i = 0; i < N_ + 1; i++) {
    const auto& E = E_(i);
    const auto& L = L_(i);

    auto xi = w.block(0, i, nx_, 1);
    auto ui = w.block(nx_, i, nu_, 1);

    const auto vi = v.col(i);
    // x(i) += a*E(i)' * v(i)
    // u(i) += a*L(i)' * v(i)
    if (a == 1.0) {
      xi.noalias() += E.transpose() * vi;
      ui.noalias() += L.transpose() * vi;
    } else if (a == -1.0) {
      xi.noalias() -= E.transpose() * vi;
      ui.noalias() -= L.transpose() * vi;
    } else {
      ui.noalias() += a * L.transpose().lazyProduct(vi);
      xi.noalias() += a * E.transpose().lazyProduct(vi);
    }
  }
}

void MpcData::axpyf(double a, Eigen::VectorXd* y) const {
  if (y == nullptr) {
    throw std::runtime_error("In MpcData::axpyf: y input is null.");
  }
  if (y->size() != nz_) {
    throw std::runtime_error("Size mismatch in MpcData::axpyf.");
  }

  // Create reshaped view of the input vector.
  Map w(y->data(), nx_ + nu_, N_ + 1);

  for (int i = 0; i < N_ + 1; i++) {
    auto xi = w.block(0, i, nx_, 1);
    auto ui = w.block(nx_, i, nu_, 1);

    xi.noalias() += a * q_(i);
    ui.noalias() += a * r_(i);
  }
}

void MpcData::axpyh(double a, Eigen::VectorXd* y) const {
  if (y == nullptr) {
    throw std::runtime_error("In MpcData::axpyh: y input is null.");
  }
  if (y->size() != nl_) {
    throw std::runtime_error("Size mismatch in MpcData::axpyh.");
  }
  // Create reshaped view of the input vector.
  Map w(y->data(), nx_, N_ + 1);
  w.col(0) += -a * x0_;

  for (int i = 1; i < N_ + 1; i++) {
    w.col(i) += -a * c_(i - 1);
  }
}

void MpcData::axpyb(double a, Eigen::VectorXd* y) const {
  if (y == nullptr) {
    throw std::runtime_error("In MpcData::axpyb: y input is null.");
  }
  if (y->size() != nv_) {
    throw std::runtime_error("Size mismatch in MpcData::axpyb.");
  }
  // Create reshaped view of the input vector.
  Map w(y->data(), nc_, N_ + 1);

  for (int i = 0; i < N_ + 1; i++) {
    w.col(i).noalias() += -a * d_(i);
  }
}

void MpcData::ValidateInputs() const {
  bool OK = true;
  const int N = Q_.length();
  if (N <= 0) {
    throw std::runtime_error("Horizon length must be at least 1.");
  }

  OK = OK && N == R_.length();
  OK = OK && N == S_.length();
  OK = OK && N == q_.length();
  OK = OK && N == r_.length();
  OK = OK && (N - 1) == A_.length();
  OK = OK && (N - 1) == B_.length();
  OK = OK && (N - 1) == c_.length();
  OK = OK && N == E_.length();
  OK = OK && N == L_.length();
  OK = OK && N == d_.length();
  if (!OK) {
    throw std::runtime_error(
        "Sequence length mismatch in input data to MpcData.");
  }

  const int nx = Q_.rows();
  if (x0_.size() != nx) {
    throw std::runtime_error("Size mismatch in x0 input to MpcData.");
  }
  if (Q_.cols() != nx) {
    throw std::runtime_error("Size mismatch in Q input to MpcData.");
  }
  if (S_.cols() != nx) {
    throw std::runtime_error("Size mismatch in S input to MpcData.");
  }
  if (q_.rows() != nx) {
    throw std::runtime_error("Size mismatch in q input to MpcData.");
  }
  if (E_.cols() != nx) {
    throw std::runtime_error("Size mismatch in E input to MpcData.");
  }
  if (A_.rows() != nx || A_.cols() != nx) {
    throw std::runtime_error("Size mismatch in A input to MpcData.");
  }
  if (B_.rows() != nx) {
    throw std::runtime_error("Size mismatch in B input to MpcData.");
  }
  if (c_.rows() != nx) {
    throw std::runtime_error("Size mismatch in c input to MpcData.");
  }

  const int nu = R_.rows();
  if (R_.cols() != nu) {
    throw std::runtime_error("Size mismatch in R input to MpcData.");
  }
  if (S_.rows() != nu) {
    throw std::runtime_error("Size mismatch in S input to MpcData.");
  }
  if (r_.rows() != nu) {
    throw std::runtime_error("Size mismatch in r input to MpcData.");
  }
  if (L_.cols() != nu) {
    throw std::runtime_error("Size mismatch in L input to MpcData.");
  }
  if (B_.cols() != nu) {
    throw std::runtime_error("Size mismatch in B input to MpcData.");
  }

  const int nc = E_.rows();
  if (L_.rows() != nc) {
    throw std::runtime_error("Size mismatch in L input to MpcData.");
  }
  if (d_.rows() != nc) {
    throw std::runtime_error("Size mismatch in d input to MpcData.");
  }
}

}  // namespace fbstab
