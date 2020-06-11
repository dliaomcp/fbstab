#pragma once

#include <Eigen/Dense>

namespace fbstab {

/**
 * Stores data and provides methods for working with quadratic programs of the
 * form (1)
 *
 *     min.  1/2 z'Hz + f'z
 *     s.t.  Gz =  h
 *           Az <= b
 *
 * Its KKT system (2) is
 *
 *     Hz + f + G' l + A' v = 0
 *     h - Gz = 0
 *     b - Az >= 0, v >= 0
 *     (b - Az)' v = 0
 *
 * where l and v are dual variables.
 */
class Data {
 public:
  virtual ~Data() = 0;

  /** Performs the operation y <- a*H*x + b*y */
  virtual void gemvH(const Eigen::VectorXd& x, double a, double b,
                     Eigen::VectorXd* y) const = 0;

  /** Performs the operation y <- a*A*x + b*y */
  virtual void gemvA(const Eigen::VectorXd& x, double a, double b,
                     Eigen::VectorXd* y) const = 0;

  /** Performs the operation y <- a*G*x + b*y */
  virtual void gemvG(const Eigen::VectorXd& x, double a, double b,
                     Eigen::VectorXd* y) const = 0;

  /** Performs the operation y <- a*A'*x + b*y */
  virtual void gemvAT(const Eigen::VectorXd& x, double a, double b,
                      Eigen::VectorXd* y) const = 0;

  /** Performs the operation y <- a*G'*x + b*y */
  virtual void gemvGT(const Eigen::VectorXd& x, double a, double b,
                      Eigen::VectorXd* y) const = 0;

  /** Performs the operation y <- a*f + y */
  virtual void axpyf(double a, Eigen::VectorXd* y) const = 0;

  /** Performs the operation y <- a*h + y */
  virtual void axpyh(double a, Eigen::VectorXd* y) const = 0;

  /** Performs the operation y <- a*b + y */
  virtual void axpyb(double a, Eigen::VectorXd* y) const = 0;

  /**
   * Norm of the forcing term.
   * @return ||(f,h,b)||
   */
  virtual double ForcingNorm() const = 0;
};

inline Data::~Data(){};

/**
 * Represents primal-dual variables for (1) and provides methods for operating
 * on them. Variables have 4 fields:
 * - z: Decision variables
 * - l: Equality duals
 * - v: Inequality duals
 * - y: Inequality margins
 */
template <class Derived>
class Variable {
 public:
  virtual ~Variable() = 0;

  /**
   * Links to problem data needed to perform calculations.
   * Calculations cannot be performed until a data object is provided.
   * @param[in] data pointer to the problem data
   */
  virtual void LinkData(const Data* data) = 0;

  /**
   * Fills the variable with one value.
   * @param[in] a
   */
  virtual void Fill(double a) = 0;

  /**
   * Sets the constraint margin to y = b - Az.
   * Throws a runtime_error if problem data hasn't been provided.
   */
  virtual void InitializeConstraintMargin() = 0;

  /**
   * Performs the operation *this <- a*x + *this.
   *
   * @param[in] a scalar
   * @param[in] x vector
   *
   * Throws a runtime_error if problem data hasn't been provided.
   */
  virtual void axpy(double a, const Derived& x) = 0;

  /**
   * Deep copies x into *this.
   * @param[in] x variable to be copied.
   */
  virtual void Copy(const Derived& x) = 0;

  /**
   * Projects the inequality duals onto the non-negative orthant,
   * i.e., v <- max(0,v).
   */
  virtual void ProjectDuals() = 0;

  /**
   * Computes the Euclidean norm.
   * @return sqrt(|z|^2 + |l|^2 + |v|^2)
   */
  virtual double Norm() const = 0;
};

template <class Derived>
inline Variable<Derived>::~Variable(){};

/**
 * This class computes and stores residuals for various reformulations of the
 * KKT system (2).
 *
 * Residuals have 3 components:
 * - z: Stationarity residual
 * - l: Equality residual
 * - v: Inequality/complimentarity residual
 */
template <class Derived, class Variable>
class Residual {
 public:
  virtual ~Residual() = 0;

  /**
   * Sets the value of alpha used in residual computations,
   * see (19) in https://arxiv.org/pdf/1901.04046.pdf.
   * @param[in] alpha
   */
  virtual void SetAlpha(double alpha) = 0;

  /**
   * Links to problem data needed to perform calculations.
   * Calculations cannot be performed until a data object is provided.
   * @param[in] data pointer to the problem data
   */
  virtual void LinkData(const Data* data) = 0;

  /**
   * Fills storage with a.
   * @param[in] a
   */
  virtual void Fill(double a) = 0;

  /**
   * Sets *this <- -1* *this.
   */
  virtual void Negate() = 0;

  /**
   * Euclidean norm of the residuals.
   * @return sqrt(z^2 + l^2 + v^2)
   */
  virtual double Norm() const = 0;

  /**
   * @return 0.5*Norm()^2
   */
  virtual double Merit() const = 0;

  /**
   * Computes R(x,xbar,sigma), the residual of a proximal subproblem
   * and stores the result internally.
   * R(x,xbar,sigma) = 0 if and only if x = P(xbar,sigma)
   * where P is the proximal operator.
   *
   * See (11) and (20) in https://arxiv.org/pdf/1901.04046.pdf
   * for a mathematical description.
   *
   * @param[in] x      Inner loop variable
   * @param[in] xbar   Outer loop variable
   * @param[in] sigma  Regularization strength > 0
   *
   * Throws a runtime_error if sigma isn't positive,
   * or if x and xbar aren't the same size.
   */
  virtual void InnerResidual(const Variable& x, const Variable& xbar,
                             double sigma) = 0;

  /**
   * Computes π(x): the natural residual of the QP
   * at the primal-dual point x and stores the result internally.
   * See (17) in https://arxiv.org/pdf/1901.04046.pdf
   * for a mathematical definition.
   *
   * @param[in] x Evaluation point.
   */
  virtual void NaturalResidual(const Variable& x);

  /**
   * Computes the natural residual function augmented with
   * penalty terms, it is analogous to (18) in
   * https://arxiv.org/pdf/1901.04046.pdf,
   * and stores the result internally.
   *
   * @param[in] x Evaluation point.
   */
  virtual void PenalizedNaturalResidual(const Variable& x);

  /** @return ||z|| */
  virtual double z_norm() const = 0;
  /** @return ||l|| */
  virtual double l_norm() const = 0;
  /** @return ||v|| */
  virtual double v_norm() const = 0;
};

template <class Derived, class Variable>
inline Residual<Derived, Variable>::~Residual(){};

/**
 * This class detects infeasibility in (2).
 * It contains methods for determining if a primal-dual variable
 * is a certificate of either unboundedness (dual infeasibility)
 * or primal infeasibility. It implements
 * Algorithm 3 of https://arxiv.org/pdf/1901.04046.pdf.
 */
template <class Variable>
class FeasibilityResidual {
 public:
  // Codes for infeasibility detection.
  enum class FeasibilityStatus {
    FEASIBLE = 0,
    PRIMAL_INFEASIBLE = 1,
    DUAL_INFEASIBLE = 2,
    BOTH = 3
  };

  virtual ~FeasibilityResidual() = 0;

  /**
   * Links to problem data needed to perform calculations.
   * Calculations cannot be performed until a data object is provided.
   * @param[in] data pointer to the problem data
   */
  virtual void LinkData(const Data* data) = 0;

  /**
   * Checks to see if x is an infeasibility certificate for the QP and stores
   * the result internally.
   * @param[in] x   infeasibility certificate candidate
   * @param[in] tol numerical tolerance
   *
   * @return    FeasibilityStatus
   *
   * Throws a runtime_error if x and *this aren't the same size
   * or if the problem data hasn't been linked.
   */
  virtual FeasibilityStatus CheckFeasibility(const Variable& x, double tol) = 0;
};

template <class Variable>
inline FeasibilityResidual<Variable>::~FeasibilityResidual(){};

/**
 * Solves linear systems of equations for Newton steps on (2).
 * The equations are of the form
 *
 *     [Hs  G' A'][dz] = [rz]
 *     [-G  sI 0 ][dl] = [rl]
 *     [-CA 0  D ][dv] = [rv]
 *
 * where s = sigma, C = diag(gamma), D = diag(mu + sigma*gamma).
 * The vectors gamma and mu are defined in (24) of
 * https://arxiv.org/pdf/1901.04046.pdf.
 *
 * In compact form:
 *
 *     V(x,xbar,sigma)*dx = r.
 *
 */
template <class Variable, class Residual, class Data>
class LinearSolver {
 public:
  virtual ~LinearSolver() = 0;

  /**
   * Sets a parameter used in the algorithm, see (19)
   * in https://arxiv.org/pdf/1901.04046.pdf.
   * @param[in] alpha
   */
  virtual void SetAlpha(double alpha) = 0;

  /**
   * Links to problem data needed to perform calculations.
   * Calculations cannot be performed until a data object is provided.
   * @param[in] data pointer to the problem data
   */
  virtual void LinkData(const Data* data) = 0;

  /**
   * Prepares to solve V(x,xbar,s) dx = r.
   * For a direct method this usually involves factorization, for an iterative
   * method it usually means preparing a preconditioner.
   *
   * The matrix V is computed as described in
   * Algorithm 4 of https://arxiv.org/pdf/1901.04046.pdf.
   *
   * @param[in]  x       Inner loop iterate
   * @param[in]  xbar    Outer loop iterate
   * @param[in]  sigma   Regularization strength
   * @return             true if successful
   */
  virtual bool Initialize(const Variable& x, const Variable& y, double s) = 0;

  /**
   * Solves the system V*x = r and stores the result in x.
   * This method assumes that the Initialize routine was run first,
   *
   * @param[in]   r   The right hand side vector
   * @param[out]  dx  Overwritten with the solution
   * @return          true if successful
   */
  virtual bool Solve(const Residual& r, Variable* dx) const = 0;

  // TODO: Add Multiply function that computes y = V*x,
  // to enable iterative refinement
  // virtual void Multiply(const Variable& x, Variable* y) const = 0;
};

template <class Variable, class Residual, class Data>
inline LinearSolver<Variable, Residual, Data>::~LinearSolver(){};

}  // namespace fbstab