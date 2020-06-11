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
#include "tools/matrix_sequence.h"
#include "tools/utilities.h"

namespace fbstab {

FBstabMpc::ProblemDataRef::ProblemDataRef(
    const MatrixSequence* Q_, const MatrixSequence* R_,
    const MatrixSequence* S_, const MatrixSequence* q_,
    const MatrixSequence* r_, const MatrixSequence* A_,
    const MatrixSequence* B_, const MatrixSequence* c_,
    const MatrixSequence* E_, const MatrixSequence* L_,
    const MatrixSequence* d_, const Eigen::VectorXd* x0_)
    : Q(*Q_),
      R(*R_),
      S(*S_),
      q(*q_),
      r(*r_),
      A(*A_),
      B(*B_),
      c(*c_),
      E(*E_),
      L(*L_),
      d(*d_),
      x0(x0_->data(), x0_->size()) {}

FBstabMpc::Variable::Variable(int N, int nx, int nu, int nc) {
  z = Eigen::VectorXd::Zero((N + 1) * (nx + nu));
  l = Eigen::VectorXd::Zero((N + 1) * nx);
  v = Eigen::VectorXd::Zero((N + 1) * nc);
  y = Eigen::VectorXd::Zero((N + 1) * nc);
}

FBstabMpc::Variable::Variable(const Eigen::Vector4d& s)
    : Variable(s(0), s(1), s(2), s(3)) {}

FBstabMpc::VariableRef::VariableRef(Eigen::Map<Eigen::VectorXd> z_,
                                    Eigen::Map<Eigen::VectorXd> l_,
                                    Eigen::Map<Eigen::VectorXd> v_,
                                    Eigen::Map<Eigen::VectorXd> y_)
    : z(z_), l(l_), v(v_), y(y_) {}

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

  algorithm_ = tools::make_unique<Algorithm>(
      x1_.get(), x2_.get(), x3_.get(), x4_.get(), r1_.get(), r2_.get(),
      linear_solver_.get(), feasibility_checker_.get());

  opts_ = DefaultOptions();
}

FBstabMpc::FBstabMpc(const Eigen::Vector4d& s)
    : FBstabMpc(s(0), s(1), s(2), s(3)) {}

void FBstabMpc::UpdateOptions(const Options& options) {
  // No need to validate since there are no additional options and the
  // algorithm will check the algorithmic ones.
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
template class FBstabAlgorithm<FullVariable, FullResidual, RiccatiLinearSolver,
                               FullFeasibility>;

}  // namespace fbstab
