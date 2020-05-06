#include "fbstab/fbstab_mpc.h"

#include <Eigen/Dense>
#include <memory>
#include <stdexcept>
#include <vector>

#include "fbstab/components/full_feasibility.h"
#include "fbstab/components/full_residual.h"
#include "fbstab/components/full_variable.h"
#include "fbstab/components/mpc_data.h"
#include "fbstab/components/riccati_linear_solver.h"
#include "tools/utilities.h"

namespace fbstab {

void FBstabMpc::Variable::initialize(int N, int nx, int nu, int nc) {
  z = Eigen::VectorXd::Zero((N + 1) * (nx + nu));
  l = Eigen::VectorXd::Zero((N + 1) * nx);
  v = Eigen::VectorXd::Zero((N + 1) * nc);
  y = Eigen::VectorXd::Zero((N + 1) * nc);
}

FBstabMpc::VariableRef::VariableRef(int nz, int nl, int nv, double* z_,
                                    double* l_, double* v_, double* y_)
    : z(z_, nz), l(l_, nl), v(v_, nv), y(y_, nv) {}

FBstabMpc::VariableRef::VariableRef(Map z_, Map l_, Map v_, Map y_)
    : z(z_.data(), z_.size()),
      l(l_.data(), l_.size()),
      v(v_.data(), v_.size()),
      y(y_.data(), y_.size()) {}

void FBstabMpc::VariableRef::fill(double a) {
  z.fill(a);
  l.fill(a);
  v.fill(a);
  y.fill(a);
}

FBstabMpc::FBstabMpc(int N, int nx, int nu, int nc) {
  if (N < 1 || nx < 1 || nu < 1 || nc < 1) {
    throw std::runtime_error(
        "In FBstabMpc::FBstabMpc: problem sizes must be positive.");
  }

  N_ = N;
  nx_ = nx;
  nu_ = nu;
  nc_ = nc;
  nz_ = (nx + nu) * (N + 1);
  nl_ = nx * (N + 1);
  nv_ = nc * (N + 1);

  // create the components
  x1_ = tools::make_unique<FullVariable>(nz_, nl_, nv_);
  x2_ = tools::make_unique<FullVariable>(nz_, nl_, nv_);
  x3_ = tools::make_unique<FullVariable>(nz_, nl_, nv_);
  x4_ = tools::make_unique<FullVariable>(nz_, nl_, nv_);
  r1_ = tools::make_unique<FullResidual>(nz_, nl_, nv_);
  r2_ = tools::make_unique<FullResidual>(nz_, nl_, nv_);
  feasibility_checker_ = tools::make_unique<FullFeasibility>(nz_, nl_, nv_);
  linear_solver_ = tools::make_unique<RiccatiLinearSolver>(N, nx, nu, nc);

  algorithm_ = tools::make_unique<FBstabAlgoMpc>(
      x1_.get(), x2_.get(), x3_.get(), x4_.get(), r1_.get(), r2_.get(),
      linear_solver_.get(), feasibility_checker_.get());

  opts_ = DefaultOptions();
}

FBstabMpc::FBstabMpc(const Eigen::Vector4d& s)
    : FBstabMpc(s(0), s(1), s(2), s(3)) {}

template <class InputVector>
SolverOut FBstabMpc::Solve(const ProblemData& qp, InputVector* x) {
  // The data object performs its own validation checks.
  MpcData data(&qp.Q, &qp.R, &qp.S, &qp.q, &qp.r, &qp.A, &qp.B, &qp.c, &qp.E,
               &qp.L, &qp.d, &qp.x0);
  ValidateInputSizes(data, *x);
  return algorithm_->Solve(&data, &x->z, &x->l, &x->v, &x->y);
}

void FBstabMpc::UpdateOptions(const Options& options) {
  // No need to validate since there are no additional options and the algorithm
  // will check the algorithmic ones.
  algorithm_->UpdateParameters(&options);
}

FBstabMpc::Options FBstabMpc::DefaultOptions() {
  Options opts;
  opts.DefaultParameters();
  return opts;
}

FBstabMpc::Options FBstabMpc::ReliableOptions() {
  Options opts;
  opts.ReliableParameters();
  return opts;
}

template <class InputVector>
void FBstabMpc::ValidateInputSizes(const MpcData& data, const InputVector& x) {
  if (data.N() != N_ || data.nx() != nx_ || data.nu() != nu_ ||
      data.nc() != nc_) {
    throw std::runtime_error(
        "In FBstabMpc::Solve: mismatch between *this and data dimensions.");
  }
  if (x.z.size() != nz_ || x.l.size() != nl_ || x.v.size() != nv_ ||
      x.y.size() != nv_) {
    throw std::runtime_error(
        "In FBstabMpc::Solve: mismatch between *this and initial guess "
        "dimensions.");
  }
}

// Explicit instantiation.
template class FBstabAlgorithm<FullVariable, FullResidual, MpcData,
                               RiccatiLinearSolver, FullFeasibility>;

template SolverOut FBstabMpc::Solve(const ProblemData& qp, Variable* x);
template SolverOut FBstabMpc::Solve(const ProblemData& qp, VariableRef* x);

}  // namespace fbstab
