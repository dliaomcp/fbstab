
/**
 * This file implements fbstab_algorithm.h and is included there.
 * It should never be included directly.
 */

inline void AlgorithmParameters::ValidateOptions() {
  sigma0 = std::max(sigma0, 1e-10);
  sigma_max = tools::saturate(sigma_max, 1e-6, 1e2);
  sigma_min = tools::saturate(sigma_min, 1e-13, 1e-8);
  sigma0 = tools::saturate(sigma0, sigma_min, sigma_max);

  alpha = tools::saturate(alpha, 0.001, 0.999);
  beta = tools::saturate(beta, 0.1, 0.99);
  eta = tools::saturate(eta, 1e-12, 0.499);
  delta = tools::saturate(delta, 0.0001, 0.99);
  gamma = tools::saturate(gamma, 0.001, 0.9);

  abs_tol = std::max(abs_tol, 1e-14);
  rel_tol = std::max(rel_tol, 0.0);
  stall_tol = std::max(stall_tol, 1e-14);
  infeas_tol = std::max(infeas_tol, 1e-14);

  inner_tol_max = tools::saturate(inner_tol_max, 1e-8, 1e2);
  inner_tol_min = tools::saturate(inner_tol_min, 1e-14, 1e-2);

  max_newton_iters = std::max(max_newton_iters, 1);
  max_prox_iters = std::max(max_prox_iters, 1);
  max_inner_iters = std::max(max_inner_iters, 1);
  max_linesearch_iters = std::max(max_linesearch_iters, 1);
}

inline void AlgorithmParameters::DefaultParameters() {
  sigma0 = 1e-8;
  sigma_max = 1e-6;
  sigma_min = 1e-12;
  alpha = 0.95;
  beta = 0.75;
  eta = 1e-8;
  delta = 0.2;
  gamma = 0.1;

  abs_tol = 1e-6;
  rel_tol = 1e-12;
  stall_tol = 1e-10;
  infeas_tol = 1e-8;

  inner_tol_max = 1e-2;
  inner_tol_min = 1e-12;

  max_newton_iters = 200;
  max_prox_iters = 30;
  max_inner_iters = 50;
  max_linesearch_iters = 20;

  check_feasibility = true;
  nonmonotone_linesearch = true;
  display_level = Display::FINAL;
}

inline void AlgorithmParameters::ReliableParameters() {
  DefaultParameters();
  sigma0 = 1e-4;
  sigma_max = 1e-2;
  sigma_min = 1e-10;
  beta = 0.9;

  abs_tol = 1e-4;
  rel_tol = 1e-6;
  max_linesearch_iters = 40;
  max_newton_iters = 500;
  max_prox_iters = 100;
  nonmonotone_linesearch = false;
}

// Constructor implementation *************************************
template <class Variable, class Residual, class LinearSolver, class Feasibility>
FBstabAlgorithm<Variable, Residual, LinearSolver, Feasibility>::FBstabAlgorithm(
    Variable* x1, Variable* x2, Variable* x3, Variable* x4, Residual* r1,
    Residual* r2, LinearSolver* lin_sol, Feasibility* fcheck) {
  if (x1 == nullptr || x2 == nullptr || x3 == nullptr || x4 == nullptr) {
    throw std::runtime_error("A Variable supplied to FBstabAlgorithm is null.");
  }
  if (r1 == nullptr || r2 == nullptr) {
    throw std::runtime_error("A Residual supplied to FBstabAlgorithm is null");
  }
  if (lin_sol == nullptr) {
    throw std::runtime_error(
        "The LinearSolver supplied to FBstabAlgorithm is null.");
  }
  if (fcheck == nullptr) {
    throw std::runtime_error(
        "The Feasibility object supplied to FBstabAlgorithm is null.");
  }

  xk_ = x1;
  xi_ = x2;
  xp_ = x3;
  dx_ = x4;

  rk_ = r1;
  ri_ = r2;

  linear_solver_ = lin_sol;
  feasibility_ = fcheck;

  opts_.DefaultParameters();
}

// Solve implementation *************************************
template <class Variable, class Residual, class LinearSolver, class Feasibility>
template <class ProblemData, class InputVector, class OutputStream>
SolverOut FBstabAlgorithm<Variable, Residual, LinearSolver, Feasibility>::Solve(
    const ProblemData& qp_data, InputVector* z0, InputVector* l0,
    InputVector* v0, InputVector* y0, const OutputStream& os) {
  const time_point start_time{clock::now() /* dummy */};

  // Make sure the linear solver and residuals objects are using the same value
  // for the alpha parameter.
  rk_->SetAlpha(opts_.alpha);
  ri_->SetAlpha(opts_.alpha);
  linear_solver_->SetAlpha(opts_.alpha);

  // Supply a pointer to the data object.
  xk_->LinkData(&qp_data);
  xi_->LinkData(&qp_data);
  dx_->LinkData(&qp_data);
  xp_->LinkData(&qp_data);

  rk_->LinkData(&qp_data);
  ri_->LinkData(&qp_data);
  feasibility_->LinkData(&qp_data);
  linear_solver_->LinkData(&qp_data);

  // Initialization phase.
  const double sigma = opts_.sigma0;
  combo_tol_ = opts_.abs_tol + opts_.rel_tol * (1.0 + qp_data.ForcingNorm());

  // Copy the initial guess into xk
  CopyIntoVariable(*z0, *l0, *v0, *y0, xk_);
  xi_->Copy(*xk_);
  dx_->Fill(1.0);

  rk_->PenalizedNaturalResidual(*xk_);
  ri_->Fill(0.0);
  double E0 = rk_->Norm();
  double Ek = E0;
  double inner_tol =
      tools::saturate(E0, opts_.inner_tol_min, opts_.inner_tol_max);

  // Reset iteration count.
  newton_iters_ = 0;
  prox_iters_ = 0;

  PrintIterHeader(os);

  // Main proximal loop.
  for (int k = 0; k < opts_.max_prox_iters; k++) {
    // The solver stops if:
    // a) ||rk|| <= abs_tol + rel_tol*(1 + ||w||)
    // b) ||x(k) - x(k-1)|| <= stall_tol
    rk_->PenalizedNaturalResidual(*xk_);
    Ek = rk_->Norm();
    if (Ek <= combo_tol_ || dx_->Norm() <= opts_.stall_tol) {
      PrintIterLine(prox_iters_, newton_iters_, *rk_, *ri_, inner_tol, os);
      SolverOut output = PrepareOutput(ExitFlag::SUCCESS, prox_iters_,
                                       newton_iters_, *rk_, start_time, E0, os);
      WriteVariable(*xk_, z0, l0, v0, y0);
      return output;
    } else {
      PrintDetailedHeader(prox_iters_, newton_iters_, *rk_, os);
      PrintIterLine(prox_iters_, newton_iters_, *rk_, *ri_, inner_tol, os);
    }

    // TODO(dliaomcp@umich.edu) Check if the residual is decreasing.
    // TODO(dliaomcp@umich.edu) Implement adaptive rule for decreasing sigma.

    // Update subproblem tolerance.
    inner_tol =
        tools::saturate(inner_tol * opts_.delta, opts_.inner_tol_min, Ek);

    // Solve the proximal subproblem.
    xi_->Copy(*xk_);
    const double Eo =
        SolveProximalSubproblem(xi_, xk_, inner_tol, sigma, Ek, os);

    // Iteration timeout check.
    if (newton_iters_ >= opts_.max_newton_iters) {
      if (Eo < Ek) {
        WriteVariable(*xi_, z0, l0, v0, y0);
        rk_->PenalizedNaturalResidual(*xi_);
      } else {
        WriteVariable(*xk_, z0, l0, v0, y0);
        rk_->PenalizedNaturalResidual(*xk_);
      }
      SolverOut output = PrepareOutput(ExitFlag::MAXITERATIONS, prox_iters_,
                                       newton_iters_, *rk_, start_time, E0, os);
      return output;
    }

    // Compute dx <- x(k+1) - x(k).
    dx_->Copy(*xi_);
    dx_->axpy(-1.0, *xk_);
    if (opts_.check_feasibility) {
      ExitFlag eflag = CheckForInfeasibility(*dx_);
      if (eflag != ExitFlag::SUCCESS) {
        SolverOut output = PrepareOutput(eflag, prox_iters_, newton_iters_,
                                         *rk_, start_time, E0, os);
        WriteVariable(*dx_, z0, l0, v0, y0);
        return output;
      }
    }

    // x(k+1) = x(i)
    xk_->Copy(*xi_);
    prox_iters_++;
  }  // end proximal loop

  // Timeout exit.
  SolverOut output = PrepareOutput(ExitFlag::MAXITERATIONS, prox_iters_,
                                   newton_iters_, *rk_, start_time, E0, os);
  WriteVariable(*xk_, z0, l0, v0, y0);
  return output;
}

// Subproblem solver implementation *************************************
template <class Variable, class Residual, class LinearSolver, class Feasibility>
template <class OutputStream>
double FBstabAlgorithm<Variable, Residual, LinearSolver, Feasibility>::
    SolveProximalSubproblem(Variable* x, Variable* xbar, double tol,
                            double sigma, double current_outer_residual,
                            const OutputStream& os) {
  merit_buffer_.fill(0.0);  // Clear the buffer of past merit function values.

  double Eo = 0;   // KKT residual.
  double t = 1.0;  // Linesearch parameter.
  for (int i = 0; i < opts_.max_inner_iters; i++) {
    // Compute subproblem residual.
    ri_->InnerResidual(*x, *xbar, sigma);
    double Ei = ri_->Norm();
    // Compute KKT residual.
    rk_->PenalizedNaturalResidual(*x);
    Eo = rk_->Norm();

    // The inner loop stops if:
    // a) The subproblem is solved to the prescribed
    // tolerance and the outer residual has been reduced.
    // b) The outer residual cannot be decreased
    // (this can happen if the problem is infeasible).
    if ((Ei <= tol && Eo < current_outer_residual) ||
        (Ei <= opts_.inner_tol_min)) {
      PrintDetailedLine(i, t, *ri_, os);
      PrintDetailedFooter(tol, *ri_, os);
      break;
    } else {
      PrintDetailedLine(i, t, *ri_, os);
    }
    if (newton_iters_ >= opts_.max_newton_iters) {
      break;
    }

    // Solve for the Newton step.
    const bool initialize_flag = linear_solver_->Initialize(*x, *xbar, sigma);
    if (initialize_flag != true) {
      throw std::runtime_error(
          "In FBstabAlgorithm::Solve: LinearSolver::Initialize failed.");
    }
    ri_->Negate();

    const bool solve_flag = linear_solver_->Solve(*ri_, dx_);
    if (solve_flag != true) {
      throw std::runtime_error(
          "In FBstabAlgorithm::Solve: LinearSolver::Solve failed.");
    }
    newton_iters_++;

    // Linesearch
    const double current_merit = ri_->Merit();
    InsertMerit(current_merit);
    const double m0 = opts_.nonmonotone_linesearch ? MaxMerit() : current_merit;

    t = 1.0;
    for (int j = 0; j < opts_.max_linesearch_iters; j++) {
      // Compute a trial point xp = x + t*dx
      // and evaluate the merit function at xp.
      xp_->Copy(*x);
      xp_->axpy(t, *dx_);
      ri_->InnerResidual(*xp_, *xbar, sigma);
      const double mp = ri_->Merit();

      // Armijo descent check.
      if (mp <= m0 - 2.0 * t * opts_.eta * current_merit) {
        break;
      } else {
        t *= opts_.beta;
      }
    }
    x->axpy(t, *dx_);  // x <- x + t*dx
  }
  // Make duals non-negative.
  x->ProjectDuals();

  return Eo;
}

// Helper functions *************************************
template <class Variable, class Residual, class LinearSolver, class Feasibility>
void FBstabAlgorithm<Variable, Residual, LinearSolver, Feasibility>::
    UpdateParameters(const AlgorithmParameters* const options) {
  opts_.sigma0 = options->sigma0;
  opts_.sigma_min = options->sigma_min;
  opts_.sigma_max = options->sigma_max;
  opts_.alpha = options->alpha;
  opts_.beta = options->beta;
  opts_.eta = options->eta;
  opts_.delta = options->delta;
  opts_.gamma = options->gamma;
  opts_.abs_tol = options->abs_tol;
  opts_.rel_tol = options->rel_tol;
  opts_.stall_tol = options->stall_tol;
  opts_.infeas_tol = options->infeas_tol;
  opts_.inner_tol_max = options->inner_tol_max;
  opts_.inner_tol_min = options->inner_tol_min;
  opts_.max_newton_iters = options->max_newton_iters;
  opts_.max_prox_iters = options->max_prox_iters;
  opts_.max_inner_iters = options->max_inner_iters;
  opts_.max_linesearch_iters = options->max_linesearch_iters;
  opts_.check_feasibility = options->check_feasibility;
  opts_.nonmonotone_linesearch = options->nonmonotone_linesearch;
  opts_.display_level = options->display_level;
  opts_.ValidateOptions();
}

template <class Variable, class Residual, class LinearSolver, class Feasibility>
template <class InputVector>
void FBstabAlgorithm<Variable, Residual, LinearSolver,
                     Feasibility>::CopyIntoVariable(const InputVector& z,
                                                    const InputVector& l,
                                                    const InputVector& v,
                                                    const InputVector& y,
                                                    Variable* x) const {
  std::ignore = y;  // x->y will be overwritten by InitializeConstraintMargin
  x->z() = z;
  x->l() = l;
  x->v() = v;
  x->InitializeConstraintMargin();
}

template <class Variable, class Residual, class LinearSolver, class Feasibility>
template <class InputVector>
void FBstabAlgorithm<Variable, Residual, LinearSolver,
                     Feasibility>::WriteVariable(const Variable& x,
                                                 InputVector* z, InputVector* l,
                                                 InputVector* v,
                                                 InputVector* y) const {
  *z = x.z();
  *l = x.l();
  *v = x.v();
  *y = x.y();
}

template <class Variable, class Residual, class LinearSolver, class Feasibility>
template <class OutputStream>
SolverOut
FBstabAlgorithm<Variable, Residual, LinearSolver, Feasibility>::PrepareOutput(
    ExitFlag e, int prox_iters, int newton_iters, const Residual& r,
    time_point start, double initial_residual, const OutputStream& os) const {
  struct SolverOut output;

  time_point now = clock::now();
  std::chrono::duration<double> elapsed = now - start;
  output.solve_time = elapsed.count();

  output.eflag = e;
  output.residual = r.Norm();
  output.newton_iters = newton_iters;
  output.prox_iters = prox_iters;
  output.initial_residual = initial_residual;

  // Printing is in ms.
  PrintFinal(prox_iters, newton_iters, e, r, 1000.0 * output.solve_time, os);
  return output;
}

template <class Variable, class Residual, class LinearSolver, class Feasibility>
ExitFlag
FBstabAlgorithm<Variable, Residual, LinearSolver,
                Feasibility>::CheckForInfeasibility(const Variable& x) {
  typename Feasibility::FeasibilityStatus feas =
      feasibility_->CheckFeasibility(x, opts_.infeas_tol);
  if (feas == Feasibility::FeasibilityStatus::FEASIBLE) {
    return ExitFlag::SUCCESS;
  } else if (feas == Feasibility::FeasibilityStatus::PRIMAL_INFEASIBLE) {
    return ExitFlag::PRIMAL_INFEASIBLE;
  } else if (feas == Feasibility::FeasibilityStatus::DUAL_INFEASIBLE) {
    return ExitFlag::DUAL_INFEASIBLE;
  } else {
    return ExitFlag::PRIMAL_DUAL_INFEASIBLE;
  }
}

template <class Variable, class Residual, class LinearSolver, class Feasibility>
void FBstabAlgorithm<Variable, Residual, LinearSolver,
                     Feasibility>::InsertMerit(double x) {
  for (auto i = merit_buffer_.size() - 1; i > 0; i--) {
    merit_buffer_.at(i) = merit_buffer_.at(i - 1);
  }
  merit_buffer_.at(0) = x;
}

template <class Variable, class Residual, class LinearSolver, class Feasibility>
template <class OutputStream>
void FBstabAlgorithm<Variable, Residual, LinearSolver,
                     Feasibility>::PrintIterLine(int prox_iters,
                                                 int newton_iters,
                                                 const Residual& rk,
                                                 const Residual& ri,
                                                 double itol,
                                                 const OutputStream& os) const {
  char buff[100];
  if (opts_.display_level == Display::ITER) {
    snprintf(buff, 100, "%12d  %12d  %12.4e  %12.4e  %12.4e  %12.4e  %12.4e\n",
             prox_iters, newton_iters, rk.z_norm(), rk.l_norm(), rk.v_norm(),
             ri.Norm(), itol);
    os.Print(buff);
  }
}

template <class Variable, class Residual, class LinearSolver, class Feasibility>
template <class OutputStream>
void FBstabAlgorithm<Variable, Residual, LinearSolver,
                     Feasibility>::PrintIterHeader(const OutputStream& os)
    const {
  char buff[100];
  if (opts_.display_level == Display::ITER) {
    snprintf(buff, 100, "%12s  %12s  %12s  %12s  %12s  %12s  %12s\n",
             "prox iter", "newton iters", "|rz|", "|rl|", "|rv|", "Inner res",
             "Inner tol");
    os.Print(buff);
  }
}

template <class Variable, class Residual, class LinearSolver, class Feasibility>
template <class OutputStream>
void FBstabAlgorithm<Variable, Residual, LinearSolver, Feasibility>::
    PrintDetailedHeader(int prox_iters, int newton_iters, const Residual& r,
                        const OutputStream& os) const {
  char buff[100];
  if (opts_.display_level == Display::ITER_DETAILED) {
    double t = r.Norm();
    snprintf(buff, 100,
             "Begin Prox Iter: %d, Total Newton Iters: %d, Residual: %6.4e\n",
             prox_iters, newton_iters, t);
    os.Print(buff);
    snprintf(buff, 100, "%10s  %10s  %10s  %10s  %10s\n", "Iter", "Step Size",
             "|rz|", "|rl|", "|rv|");
    os.Print(buff);
  }
}

template <class Variable, class Residual, class LinearSolver, class Feasibility>
template <class OutputStream>
void FBstabAlgorithm<Variable, Residual, LinearSolver, Feasibility>::
    PrintDetailedLine(int iter, double step_length, const Residual& r,
                      const OutputStream& os) const {
  char buff[100];
  if (opts_.display_level == Display::ITER_DETAILED) {
    snprintf(buff, 100, "%10d  %10e  %10e  %10e  %10e\n", iter, step_length,
             r.z_norm(), r.l_norm(), r.v_norm());
    os.Print(buff);
  }
}

template <class Variable, class Residual, class LinearSolver, class Feasibility>
template <class OutputStream>
void FBstabAlgorithm<Variable, Residual, LinearSolver,
                     Feasibility>::PrintDetailedFooter(double tol,
                                                       const Residual& r,
                                                       const OutputStream& os)
    const {
  char buff[100];
  if (opts_.display_level == Display::ITER_DETAILED) {
    snprintf(buff, 100,
             "Exiting inner loop. Inner residual: %6.4e, Inner tolerance: "
             "%6.4e\n",
             r.Norm(), tol);
    os.Print(buff);
  }
}

template <class Variable, class Residual, class LinearSolver, class Feasibility>
template <class OutputStream>
void FBstabAlgorithm<Variable, Residual, LinearSolver, Feasibility>::PrintFinal(
    int prox_iters, int newton_iters, ExitFlag eflag, const Residual& r,
    double t, const OutputStream& os) const {
  char buff[100];
  if (opts_.display_level >= Display::FINAL) {
    snprintf(buff, 100, "\nOptimization completed!  Exit code:");
    os.Print(buff);
    switch (eflag) {
      case ExitFlag::SUCCESS:
        snprintf(buff, 100, " Success\n");
        break;
      case ExitFlag::DIVERGENCE:
        snprintf(buff, 100, " Divergence\n");
        break;
      case ExitFlag::MAXITERATIONS:
        snprintf(buff, 100, " Iteration limit exceeded\n");
        break;
      case ExitFlag::PRIMAL_INFEASIBLE:
        snprintf(buff, 100, " Primal Infeasibility\n");
        break;
      case ExitFlag::DUAL_INFEASIBLE:
        snprintf(buff, 100, " Dual Infeasibility\n");
        break;
      case ExitFlag::PRIMAL_DUAL_INFEASIBLE:
        snprintf(buff, 100, " Primal-Dual Infeasibility\n");
        break;
      default:
        std::runtime_error(" Undefined exit flag supplied to PrintFinal.\n");
    }
    os.Print(buff);
    snprintf(buff, 100,
             "Time elapsed: %f ms (-1.0 indicates timing disabled)\n", t);
    os.Print(buff);
    snprintf(buff, 100, "Proximal iterations: %d out of %d\n", prox_iters,
             opts_.max_prox_iters);
    os.Print(buff);
    snprintf(buff, 100, "Newton iterations: %d out of %d\n", newton_iters,
             opts_.max_newton_iters);
    os.Print(buff);
    snprintf(buff, 100, "%10s  %10s  %10s  %10s\n", "|rz|", "|rl|", "|rv|",
             "Tolerance");
    os.Print(buff);
    snprintf(buff, 100, "%10.4e  %10.4e  %10.4e  %10.4e\n", r.z_norm(),
             r.l_norm(), r.v_norm(), combo_tol_);
    os.Print(buff);
    snprintf(buff, 100, "\n");
    os.Print(buff);
  }
}
