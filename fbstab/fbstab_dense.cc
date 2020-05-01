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
#include "tools/utilities.h"

namespace fbstab {

FBstabDense::FBstabDense(int nz, int nv) {
  if (nz <= 0 || nv <= 0) {
    throw std::runtime_error(
        "In FBstabDense::FBstabDense: Inputs must be positive.");
  }
  nz_ = nz;
  nv_ = nv;
  nl_ = 0;

  x1_ = tools::make_unique<FullVariable>(nz_, nl_, nv_);
  x2_ = tools::make_unique<FullVariable>(nz_, nl_, nv_);
  x3_ = tools::make_unique<FullVariable>(nz_, nl_, nv_);
  x4_ = tools::make_unique<FullVariable>(nz_, nl_, nv_);

  r1_ = tools::make_unique<FullResidual>(nz_, nl_, nv_);
  r2_ = tools::make_unique<FullResidual>(nz_, nl_, nv_);

  feasibility_checker_ = tools::make_unique<FullFeasibility>(nz_, nl_, nv_);
  linear_solver_ = tools::make_unique<DenseCholeskySolver>(nz_, nv_);

  algorithm_ = tools::make_unique<FBstabAlgoDense>(
      x1_.get(), x2_.get(), x3_.get(), x4_.get(), r1_.get(), r2_.get(),
      linear_solver_.get(), feasibility_checker_.get());

  opts_ = DefaultOptions();
}

SolverOut FBstabDense::Solve(const QPData& qp, QPVariable* x,
                             bool use_initial_guess) {
  DenseData data(qp.H, qp.f, qp.A, qp.b);

  // Temporary workaround:
  FullVariable x0(x->z, &x->l, x->v, x->y);

  if (nz_ != data.nz_ || nv_ != data.nv_) {
    throw std::runtime_error(
        "In FBstabDense::Solve: mismatch between *this and data dimensions.");
  }
  if (nz_ != x0.z().size() || nv_ != x0.v().size()) {
    throw std::runtime_error(
        "In FBstabDense::Solve: mismatch between *this and initial guess "
        "dimensions.");
  }
  if (!use_initial_guess) {
    x0.Fill(0.0);
  }

  return algorithm_->Solve(&data, &x0);
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

// Explicit instantiation.
template class FBstabAlgorithm<FullVariable, FullResidual, DenseData,
                               DenseCholeskySolver, FullFeasibility>;

}  // namespace fbstab
