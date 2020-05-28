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

FBstabDense::VariableRef::VariableRef(int nz, int nl, int nv, double* z_,
                                      double* l_, double* v_, double* y_)
    : z(z_, nz), l(l_, nl), v(v_, nv), y(y_, nv) {}

FBstabDense::VariableRef::VariableRef(Map z_, Map l_, Map v_, Map y_)
    : z(z_.data(), z_.size()),
      l(l_.data(), l_.size()),
      v(v_.data(), v_.size()),
      y(y_.data(), y_.size()) {}

void FBstabDense::VariableRef::fill(double a) {
  z.fill(a);
  l.fill(a);
  v.fill(a);
  y.fill(a);
}

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

template <class InputVector>
SolverOut FBstabDense::Solve(const ProblemData& qp, InputVector* x) {
  // Data performs its own validation checks.
  DenseData data(&qp.H, &qp.f, &qp.G, &qp.h, &qp.A, &qp.b);
  ValidateInputs(data, *x);
  return algorithm_->Solve(&data, &x->z, &x->l, &x->v, &x->y);
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

template <class InputVector>
void FBstabDense::ValidateInputs(const DenseData& data, const InputVector& x) {
  if (nz_ != data.nz() || nv_ != data.nv() || nl_ != data.nl()) {
    throw std::runtime_error(
        "In FBstabDense::Solve: mismatch between *this and data dimensions.");
  }
  if (nz_ != x.z.size() || nv_ != x.v.size()) {
    throw std::runtime_error(
        "In FBstabDense::Solve: mismatch between *this and initial guess "
        "dimensions.");
  }
}

// Explicit instantiation.
template SolverOut FBstabDense::Solve(const ProblemData& qp, Variable* x);
template SolverOut FBstabDense::Solve(const ProblemData& qp, VariableRef* x);
template class FBstabAlgorithm<FullVariable, FullResidual, DenseData,
                               DenseCholeskySolver, FullFeasibility>;

}  // namespace fbstab
