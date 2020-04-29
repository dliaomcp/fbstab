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

SolverOut FBstabMpc::Solve(const QPData& qp, const QPVariable* x,
                           bool use_initial_guess) {
  MpcData data(qp.Q, qp.R, qp.S, qp.q, qp.r, qp.A, qp.B, qp.c, qp.E, qp.L, qp.d,
               qp.x0);
  FullVariable x0(x->z, x->l, x->v, x->y);

  if (data.N_ != N_ || data.nx_ != nx_ || data.nu_ != nu_ || data.nc_ != nc_) {
    throw std::runtime_error(
        "In FBstabMpc::Solve: mismatch between *this and data dimensions.");
  }
  if (x0.nz_ != nz_ || x0.nl_ != nl_ || x0.nv_ != nv_) {
    throw std::runtime_error(
        "In FBstabMpc::Solve: mismatch between *this and initial guess "
        "dimensions.");
  }

  if (!use_initial_guess) {
    x0.Fill(0.0);
  }

  return algorithm_->Solve(&data, &x0);
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

// Explicit instantiation.
template class FBstabAlgorithm<FullVariable, FullResidual, MpcData,
                               RiccatiLinearSolver, FullFeasibility>;

}  // namespace fbstab
