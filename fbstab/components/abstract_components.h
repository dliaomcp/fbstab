#pragma once

namespace fbstab {

/**
 * Stores data and provides methods for working with quadratic programs of the
 * form
 *
 *     min.  1/2 z'Hz + f'z
 *     s.t.  Gz =  h
 *           Az <= b
 */
class Data {
 public:
  virtual ~Data() = 0;
  virtual double ForcingNorm() const = 0;
};

inline Data::~Data(){};

/**
 * Represents primal-dual variables.
 */
template <class Derived, class Data>
class Variable {
 public:
  virtual ~Variable() = 0;
  virtual void LinkData(const Data* data) = 0;
  virtual void Fill(double a) = 0;
  virtual void InitializeConstraintMargin() = 0;
  virtual void axpy(double a, const Derived& x) = 0;
  virtual void Copy(const Derived& x) = 0;
  virtual void ProjectDuals() = 0;
  virtual double Norm() const = 0;
};

template <class Derived, class Data>
inline Variable<Derived, Data>::~Variable(){};

/**
 * Computes various optimality residuals.
 */
template <class Derived, class Variable>
class Residual {
 public:
  virtual ~Residual() = 0;
  virtual void SetAlpha(double alpha) = 0;
  virtual void Fill(double a) = 0;
  virtual void Negate() = 0;
  virtual double Norm() const = 0;
  virtual double Merit() const = 0;
  virtual void InnerResidual(const Variable& x, const Variable& xbar,
                             double sigma) = 0;
  virtual void NaturalResidual(const Variable& x);
  virtual void PenalizedNaturalResidual(const Variable& x);

  virtual double z_norm() const = 0;
  virtual double l_norm() const = 0;
  virtual double v_norm() const = 0;
};

template <class Derived, class Variable>
inline Residual<Derived, Variable>::~Residual(){};

/**
 * Checks infeasibility certificate candidates.
 */
template <class Variable>
class FeasibilityResidual {
 public:
  virtual ~FeasibilityResidual() = 0;
  virtual void ComputeFeasibility(const Variable& x, double tol) = 0;
  virtual bool IsDualFeasible() const = 0;
  virtual bool IsPrimalFeasible() const = 0;
};

template <class Variable>
inline FeasibilityResidual<Variable>::~FeasibilityResidual(){};

template <class Variable, class Residual>
class LinearSolver {
 public:
  virtual ~LinearSolver() = 0;
  virtual void SetAlpha(double alpha) = 0;
  virtual bool Initialize(const Variable& x, const Variable& y, double s) = 0;
  virtual bool Solve(const Residual& r, Variable* dx) const = 0;
};

template <class Variable, class Residual>
inline LinearSolver<Variable, Residual>::~LinearSolver(){};

}  // namespace fbstab