#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "tools/utilities.h"

namespace fbstab {

// Return codes for the solver.
enum class ExitFlag {
  SUCCESS = 0,
  DIVERGENCE = 1,
  MAXITERATIONS = 2,
  PRIMAL_INFEASIBLE = 3,
  DUAL_INFEASIBLE = 4,
  PRIMAL_DUAL_INFEASIBLE = 5
};

/**
 * Output Data.
 * A negative valuve for solve_time indicates that no timing data is available.
 */
struct SolverOut {
  ExitFlag eflag = ExitFlag::MAXITERATIONS;
  double residual = 0.0;
  int newton_iters = 0;
  int prox_iters = 0;
  double solve_time = 0.0;  /// CPU time in s.
  double initial_residual = 0.0;
};

/** Display settings */
enum class Display {
  OFF = 0,           ///< no display
  FINAL = 1,         ///< prints message upon completion
  ITER = 2,          ///< basic information at each outer loop iteration
  ITER_DETAILED = 3  ///< print detailed inner loop information
};

/** Stores algorithm parameters */
struct AlgorithmParameters {
  double sigma0 = 1e-8;       ///< Initial stabilization parameter
  double sigma_max = 1e-8;    ///< Maximum stabilization parameter
  double sigma_min = 1e-10;   ///< Minimum stabilization parameter
  double alpha = 0.95;        ///< Penalized FB function parameter
  double beta = 0.7;          ///< Backtracking parameter
  double eta = 1e-8;          ///< Sufficient decrease parameter
  double delta = 1.0 / 5.0;   ///< Reduction factor for subproblem tolerance
  double gamma = 1.0 / 10.0;  ///< Reduction factor for sigma

  double abs_tol = 1e-6;     ///< Absolute tolerance
  double rel_tol = 1e-12;    ///< Relative tolerance
  double stall_tol = 1e-10;  ///< Tolerance on ||dx||
  double infeas_tol = 1e-8;  ///< Relative tolerance for feasibility checking

  double inner_tol_max = 1e-1;   ///< Maximum value for the subproblem tolerance
  double inner_tol_min = 1e-12;  ///< Minimum value for the subproblem tolerance

  int max_newton_iters = 500;     ///< Maximum number of Newton iterations
  int max_prox_iters = 100;       ///< Maximum number of proximal iterations
  int max_inner_iters = 100;      ///< Maximum number of iterations that can be
                                  ///< applied to a single subproblem
  int max_linesearch_iters = 20;  ///< Maximum backtracking linesearch steps

  bool check_feasibility = true;           ///< Controls the feasibility checker
  bool nonmonotone_linesearch = true;      ///<  Controls nonmonotone linesearch
  Display display_level = Display::FINAL;  ///< Controls verbosity

  /** Checks validity of fields and overwrites if necessary. */
  void ValidateOptions();
  /** Overwrites with defaults */
  void DefaultParameters();
  /** Overwrites with parameters for hard or ill-conditioned problems */
  void ReliableParameters();
};

using clock = std::chrono::high_resolution_clock;
/**
 * This class implements the FBstab solver for
 * convex quadratic programs, see
 * https://arxiv.org/pdf/1901.04046.pdf for more details.
 *
 * FBstab tries to solve instances of the following convex QP:
 *
 *     min.  1/2 z'*H*z + f'*z
 *
 *     s.t.  Gz =  h
 *           Az <= b
 *
 * The algorithm is implemented using to abstract objects
 * representing variables, residuals etc.
 * These are template parameters for the class and
 * should be written so as to be efficient for specific classes
 * of QPs, e.g., model predictive control QPs or sparse QPs.
 *
 * The algorithm exits when: ||π(x)|| <= abs_tol + ||π(x0)|| rel_tol
 * where π is the natural residual function,
 * (17) in https://arxiv.org/pdf/1901.04046.pdf.
 *
 * @tparam Variable:      storage and methods for primal-dual variables
 * @tparam Residual:      storage and methods for QP residuals
 * @tparam Data:          QP specific data storage and operations
 * @tparam LinearSolver:  solves Newton step systems
 * @tparam Feasibility:   checks for primal-dual infeasibility
 */
template <class Variable, class Residual, class LinearSolver, class Feasibility>
class FBstabAlgorithm {
 public:
  /**
   * Constructor: saves the components objects needed by the solver.
   *
   * @param[in] x1,x2,x3,x4 Variable objects used by the solver
   * @param[in] r1,r2 Residual objects used by the solver
   * @param[in] lin_sol Linear solver used by the solver
   * @param[in] fcheck Feasibility checker used by the solver
   */
  FBstabAlgorithm(Variable* x1, Variable* x2, Variable* x3, Variable* x4,
                  Residual* r1, Residual* r2, LinearSolver* lin_sol,
                  Feasibility* fcheck);

  /**
   * Attempts to solve the QP for the given
   * data starting from the supplied initial guess.
   *
   * @param[in] qp_data problem data
   * @param[in,out] z0    primal guess, overwritten
   * @param[in,out] l0    equality dual guess, overwritten
   * @param[in,out] v0    inequality dual guess, overwritten
   * @param[out]    y0    constraint margin
   *
   * @return Details on the solver output
   */
  template <class ProblemData, class InputVector, class OutputStream>
  SolverOut Solve(const ProblemData& qp_data, InputVector* z0, InputVector* l0,
                  InputVector* v0, InputVector* y0, const OutputStream& os);

  /**
   * Allows setting of algorithm options.
   * @param[in] option New options
   */
  void UpdateParameters(const AlgorithmParameters* const options);

  /** Returns current parameters */
  AlgorithmParameters CurrentParameters() const { return opts_; }

 private:
  using time_point = clock::time_point;

  // Combined absolute and relative tolerances
  double combo_tol_ = 0.0;

  // Iteration counters.
  int newton_iters_ = 0;
  int prox_iters_ = 0;

  // Component objects
  Variable* xk_ = nullptr;  // outer loop variable
  Variable* xi_ = nullptr;  // inner loop variable
  Variable* xp_ = nullptr;  // workspace
  Variable* dx_ = nullptr;  // workspace
  Residual* rk_ = nullptr;  // outer loop residual
  Residual* ri_ = nullptr;  // inner loop residual
  LinearSolver* linear_solver_ = nullptr;
  Feasibility* feasibility_ = nullptr;

  AlgorithmParameters opts_;

  static constexpr int kNonMonotoneLineSearch = 5;
  static_assert(kNonMonotoneLineSearch > 0,
                "kNonMonotoneLineSearch must be positive");
  std::array<double, kNonMonotoneLineSearch> merit_buffer_ = {
      {0.0, 0.0, 0.0, 0.0,
       0.0}};  // This needs to be initialized with the correct number of zeros
               // to avoid a subtle compiler error.
  /*
   * Attempts to solve a proximal subproblem x = P(xbar,sigma) using
   * the semismooth Newton's method. See (11) in
   * https://arxiv.org/pdf/1901.04046.pdf.
   *
   * @param[both]  x      Initial guess, overwritten with the solution.
   * @param[in]    xbar   Current proximal (outer) iterate
   * @param[in]    tol    Desired tolerance for the inner residual
   * @param[in]    sigma  Regularization strength
   * @param[in]    Eouter Current overall problem residual
   * @return Residual for the outer problem evaluated at x
   *
   * This method uses the member variables rk_, ri_, dx_, and xp_ as workspaces.
   */
  template <class OutputStream>
  double SolveProximalSubproblem(Variable* x, Variable* xbar, double tol,
                                 double sigma, double current_outer_residual,
                                 const OutputStream& os);

  /*
   * Prepares a suitable output structure.
   *
   * @param[in] e exit flag
   * @param[in] prox_iters
   * @param[in] newton_iters
   * @param[in] r
   * @param[in] start time instant when the solve call started
   */
  template <class OutputStream>
  SolverOut PrepareOutput(ExitFlag e, int prox_iters, int newton_iters,
                          const Residual& r, time_point start,
                          double initial_residual,
                          const OutputStream& os) const;

  // Reads an initial guess into a variable.
  template <class InputVector>
  void CopyIntoVariable(const InputVector& z, const InputVector& l,
                        const InputVector& v, const InputVector& y,
                        Variable* x) const;

  // Writes a variable back into that initial guess.
  template <class InputVector>
  void WriteVariable(const Variable& x, InputVector* z, InputVector* l,
                     InputVector* v, InputVector* y) const;

  /**
   * Checks if x certifies primal or dual infeasibility.
   * @param[in]  x
   * @return ExitFlag::SUCCESS if feasible, appropriate infeasibility flag
   * otherwise
   */
  ExitFlag CheckForInfeasibility(const Variable& x);

  /**
   * Shifts all elements in merit_buffer_ up one spot then inserts at [0].
   * @param[in] x value to be inserted at merit_buffer_[0]
   */
  void InsertMerit(double x);

  // @return maximum value in merit_buffer_
  double MaxMerit() const {
    return *std::max_element(merit_buffer_.begin(), merit_buffer_.end());
  }

  // Prints a header line to stdout depending on display settings.
  template <class OutputStream>
  void PrintIterHeader(const OutputStream& os) const;

  // Prints an iteration progress line to stdout depending on display settings.
  template <class OutputStream>
  void PrintIterLine(int prox_iters, int newton_iters, const Residual& rk,
                     const Residual& ri, double itol,
                     const OutputStream& os) const;

  // Prints a detailed header line to stdout depending on display settings.
  template <class OutputStream>
  void PrintDetailedHeader(int prox_iters, int newton_iters, const Residual& r,
                           const OutputStream& os) const;

  // Prints inner loop iterations details to stdout depending on display
  // settings.
  template <class OutputStream>
  void PrintDetailedLine(int iter, double step_length, const Residual& r,
                         const OutputStream& os) const;

  // Prints a footer to stdout depending on display settings.
  template <class OutputStream>
  void PrintDetailedFooter(double tol, const Residual& r,
                           const OutputStream& os) const;

  // Prints a summary to stdout depending on display settings.
  template <class OutputStream>
  void PrintFinal(int prox_iters, int newton_iters, ExitFlag eflag,
                  const Residual& r, double t, const OutputStream& os) const;
};

#include "fbstab/fbstab_algorithm-impl.h"

}  // namespace fbstab
