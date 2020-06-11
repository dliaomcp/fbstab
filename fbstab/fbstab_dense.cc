#include "fbstab/fbstab_dense.h"

#include <Eigen/Dense>
#include <memory>
#include <stdexcept>

#include "fbstab/components/dense_cholesky_solver.h"
#include "fbstab/components/dense_data.h"
#include "fbstab/components/full_feasibility.h"
#include "fbstab/components/full_residual.h"
#include "fbstab/components/full_variable.h"
#include "fbstab/fbstab_algorithm.h"
#include "tools/output_stream.h"
#include "tools/utilities.h"

namespace fbstab {

FBstabDense::FBstabDense(int nz, int nl, int nv) {
  if (nz <= 0 || nv <= 0 || nl < 0) {
    throw std::runtime_error(
        "In FBstabDense::FBstabDense: nz and nv must be positive, nl "
        "nonnegative");
  }
  nz_ = nz;
  nl_ = nl;
  nv_ = nv;

  x1_ = tools::make_unique<FullVariable>(nz_, nl_, nv_);
  x2_ = tools::make_unique<FullVariable>(nz_, nl_, nv_);
  x3_ = tools::make_unique<FullVariable>(nz_, nl_, nv_);
  x4_ = tools::make_unique<FullVariable>(nz_, nl_, nv_);
  r1_ = tools::make_unique<FullResidual>(nz_, nl_, nv_);
  r2_ = tools::make_unique<FullResidual>(nz_, nl_, nv_);
  feasibility_checker_ = tools::make_unique<FullFeasibility>(nz_, nl_, nv_);
  linear_solver_ = tools::make_unique<DenseCholeskySolver>(nz_, nl_, nv_);

  algorithm_ = tools::make_unique<FBstabAlgoDense>(
      x1_.get(), x2_.get(), x3_.get(), x4_.get(), r1_.get(), r2_.get(),
      linear_solver_.get(), feasibility_checker_.get());

  opts_ = DefaultOptions();
}

void FBstabDense::UpdateOptions(const Options& options) {
  // No need to validate since there are no additional options and the algorithm
  // will check the algorithmic ones.
  algorithm_->UpdateParameters(&options);
}

FBstabDense::Options FBstabDense::DefaultOptions() {
  Options opts;
  opts.DefaultParameters();
  return opts;
}

FBstabDense::Options FBstabDense::ReliableOptions() {
  Options opts;
  opts.ReliableParameters();
  return opts;
}

FBstabDense::ProblemData::ProblemData(int nz, int nl, int nv) {
  H.resize(nz, nz);
  G.resize(nl, nz);
  A.resize(nv, nz);
  f.resize(nz);
  h.resize(nl);
  b.resize(nv);
}

FBstabDense::ProblemDataRef::ProblemDataRef(
    const Eigen::Map<Eigen::MatrixXd>* H_,
    const Eigen::Map<Eigen::VectorXd>* f_,
    const Eigen::Map<Eigen::MatrixXd>* G_,
    const Eigen::Map<Eigen::VectorXd>* h_,
    const Eigen::Map<Eigen::MatrixXd>* A_,
    const Eigen::Map<Eigen::VectorXd>* b_)
    : H(H_->data(), H_->rows(), H_->cols()),
      G(G_->data(), G_->rows(), G_->cols()),
      A(A_->data(), A_->rows(), A_->cols()),
      f(f_->data(), f_->size()),
      h(h_->data(), h_->size()),
      b(b_->data(), b_->size()) {}

FBstabDense::Variable::Variable(int nz, int nl, int nv) {
  z = Eigen::VectorXd::Zero(nz);
  l = Eigen::VectorXd::Zero(nl);
  v = Eigen::VectorXd::Zero(nv);
  y = Eigen::VectorXd::Zero(nv);
}

FBstabDense::VariableRef::VariableRef(Eigen::Map<Eigen::VectorXd>* z_,
                                      Eigen::Map<Eigen::VectorXd>* l_,
                                      Eigen::Map<Eigen::VectorXd>* v_,
                                      Eigen::Map<Eigen::VectorXd>* y_)
    : z(z_->data(), z_->size()),
      l(l_->data(), l_->size()),
      v(v_->data(), v_->size()),
      y(y_->data(), y_->size()) {}

void FBstabDense::VariableRef::fill(double a) {
  z.fill(a);
  l.fill(a);
  v.fill(a);
  y.fill(a);
}

// Explicit instantiation.
template class FBstabAlgorithm<FullVariable, FullResidual, DenseCholeskySolver,
                               FullFeasibility>;

}  // namespace fbstab
