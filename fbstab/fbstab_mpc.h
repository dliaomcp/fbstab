#pragma once

#include <Eigen/Dense>
#include <memory>

#include "fbstab/components/full_feasibility.h"
#include "fbstab/components/full_residual.h"
#include "fbstab/components/full_variable.h"
#include "fbstab/components/mpc_data.h"
#include "fbstab/components/riccati_linear_solver.h"
#include "fbstab/fbstab_algorithm.h"
#include "tools/copyable_macros.h"
#include "tools/matrix_sequence.h"
#include "tools/output_stream.h"

namespace fbstab {

/**
 * FBstabMpc implements the Proximally Stabilized Semismooth Method for
 * solving the following quadratic programming problem (1):
 *
 *     min.  \sum_{i=0}^N 1/2 [x(i)]' * [Q(i) S(i)'] [x(i)] + [q(i)]'*[x(i)]
 *                            [u(i)]    [S(i) R(i) ] [u(i)]   [r(i)]  [u(i)]
 *
 *     s.t.  x(i+1) = A(i)*x(i) + B(i) u(i) + c(i), i = 0 ... N-1
 *           x(0) = x0
 *           E(i)*x(i) + L(i)*u(i) + d(i) <= 0,     i = 0 ... N
 *
 * Where
 *        [ Q(i),S(i)']
 *        [ S(i),R(i) ]
 *
 * is positive semidefinite for all i \in [0,N].
 * See also (29) in https://arxiv.org/pdf/1901.04046.pdf.
 *
 * The problem is of size (N,nx,nu,nc) where:
 * - N > 0 is the horizon length
 * - nx > 0 is the number of states
 * - nu > 0 is the number of control inputs
 * - nc > 0 is the number of constraints per stage
 *
 * This is a specialization of the general form (2),
 *
 *     min.  1/2 z'Hz + f'z
 *
 *     s.t.  Gz =  h
 *           Az <= b
 *
 * which has dimensions nz = (nx + nu) * (N + 1), nl = nx * (N + 1),
 * and nv = nc * (N + 1).
 *
 * Aside from convexity there are no assumptions made about the problem.
 * This method can detect unboundedness/infeasibility
 * and exploit arbitrary initial guesses.
 */
class FBstabMpc {
 public:
  FBSTAB_NO_COPY_NO_MOVE_NO_ASSIGN(FBstabMpc)
  // Convenience typedef.
  using Algorithm = FBstabAlgorithm<FullVariable, FullResidual,
                                    RiccatiLinearSolver, FullFeasibility>;
  /**
   * Structure to hold problem data.
   * See the class documentation or (29) in
   * https://arxiv.org/pdf/1901.04046.pdf for more details.
   */
  struct ProblemData {
    ProblemData() = default;
    MatrixSequence Q;    /// N + 1 sequence of nx x nx matrices
    MatrixSequence R;    /// N + 1 sequence of nu x nu matrices
    MatrixSequence S;    /// N + 1 sequence of nu x nx matrices
    MatrixSequence q;    /// N + 1 sequence of nx x 1  matrices
    MatrixSequence r;    /// N + 1 sequence of nu x 1  matrices
    MatrixSequence A;    /// N     sequence of nx x nx matrices
    MatrixSequence B;    /// N     sequence of nx x nu matrices
    MatrixSequence c;    /// N     sequence of nx x 1  matrices
    MatrixSequence E;    /// N + 1 sequence of nc x nx matrices
    MatrixSequence L;    /// N + 1 sequence of nc x nu matrices
    MatrixSequence d;    /// N + 1 sequence of nc x 1  matrices
    Eigen::VectorXd x0;  /// nx x 1 vector
  };

  /**
   * Structure to hold references to the problem data.
   * This struct assumes that all references remain valid.
   *
   * See the class documentation or (29) in
   * https://arxiv.org/pdf/1901.04046.pdf for more details.
   */
  struct ProblemDataRef {
    /** Creates an uninitialized object. */
    ProblemDataRef() : x0(nullptr, 0) {}

    /** Set x0 using a compatible Eigen object. */
    template <class Vector>
    void SetX0(const Vector& x0_) {
      new (&x0) Eigen::Map<const Eigen::VectorXd>(x0_.data(), x0_.size());
    }

    /** Create a reference to existing objects. */
    ProblemDataRef(const MatrixSequence* Q_, const MatrixSequence* R_,
                   const MatrixSequence* S_, const MatrixSequence* q_,
                   const MatrixSequence* r_, const MatrixSequence* A_,
                   const MatrixSequence* B_, const MatrixSequence* c_,
                   const MatrixSequence* E_, const MatrixSequence* L_,
                   const MatrixSequence* d_, const Eigen::VectorXd* x0_);

    MapMatrixSequence Q;  /// N + 1 sequence of nx x nx matrices
    MapMatrixSequence R;  /// N + 1 sequence of nu x nu matrices
    MapMatrixSequence S;  /// N + 1 sequence of nu x nx matrices
    MapMatrixSequence q;  /// N + 1 sequence of nx x 1  matrices
    MapMatrixSequence r;  /// N + 1 sequence of nu x 1  matrices
    MapMatrixSequence A;  /// N     sequence of nx x nx matrices
    MapMatrixSequence B;  /// N     sequence of nx x nu matrices
    MapMatrixSequence c;  /// N     sequence of nx x 1  matrices
    MapMatrixSequence E;  /// N + 1 sequence of nc x nx matrices
    MapMatrixSequence L;  /// N + 1 sequence of nc x nu matrices
    MapMatrixSequence d;  /// N + 1 sequence of nc x 1  matrices
    Eigen::Map<const Eigen::VectorXd> x0;  /// nx x 1 vector
  };

  /**
   * Structure to hold the initial guess and solution.
   * These vectors will be overwritten by the solve routine.
   */
  struct Variable {
    // Initialize variables to 0 for a given problem size.
    Variable(int N, int nx, int nu, int nc);
    // Initialization in vector form, s = (N, nx, nu, nc)
    Variable(const Eigen::Vector4d& s);

    Eigen::VectorXd z;  /// Decision variables in \reals^nz
    Eigen::VectorXd l;  /// Equality duals/costates in \reals^nl
    Eigen::VectorXd v;  /// Inequality duals in \reals^nv
    Eigen::VectorXd y;  /// Constraint margin, i.e., y = b-Az, in \reals^nv
  };

  /** A structure to store the solution using existing memory. */
  struct VariableRef {
    VariableRef(Eigen::Map<Eigen::VectorXd> z_, Eigen::Map<Eigen::VectorXd> l_,
                Eigen::Map<Eigen::VectorXd> v_, Eigen::Map<Eigen::VectorXd> y_);

    // Fill all fields with a.
    void fill(double a);

    Eigen::Map<Eigen::VectorXd> z;  /// Decision variables in R^nz.
    Eigen::Map<Eigen::VectorXd> l;  /// Equality duals in R^nl
    Eigen::Map<Eigen::VectorXd> v;  /// Inequality duals in R^nv.
    Eigen::Map<Eigen::VectorXd> y;  /// Constraint margin y = b-Az, in R^nv.
  };

  /** A Structure to hold options */
  struct Options : public AlgorithmParameters {};

  /**
   * Allocates workspaces needed when solving (1).
   *
   * @param[in] N Horizon length
   * @param[in] nx number of states
   * @param[in] nu number of control input
   * @param[in] nc number of constraints per timestep
   *
   * Throws a runtime_error if any inputs are nonpositive.
   */
  FBstabMpc(int N, int nx, int nu, int nc);

  // Allocates workspace with s = (N, nx, nu, nc)
  FBstabMpc(const Eigen::Vector4d& s);

  /**
   * Solves an instance of (1).
   *
   * @param[in]     qp problem data
   * @param[in,out] x  initial guess, overwritten with the solution
   * @return       Summary of the optimizer output, see fbstab_algorithm.h.
   *
   * @tparam InputData      ProblemData or ProblemDataRef
   * @tparam InputVariable  Variable or VariableRef
   * @tparam OutStream      Allows various printing classes
   */
  template <class InputData, class InputVariable, class OutStream>
  SolverOut Solve(const InputData& qp, InputVariable* x, const OutStream& os) {
    // The data object performs its own validation checks.
    MpcData data(&qp.Q, &qp.R, &qp.S, &qp.q, &qp.r, &qp.A, &qp.B, &qp.c, &qp.E,
                 &qp.L, &qp.d, &qp.x0);
    ValidateInputSizes(data, *x);
    return algorithm_->Solve(data, &x->z, &x->l, &x->v, &x->y, os);
  }

  /** Uses a default printer */
  template <class InputData, class InputVariable>
  SolverOut Solve(const InputData& qp, InputVariable* x) {
    StandardOutput os;
    return Solve(qp, x, os);
  }

  /**
   * Allows for setting of solver options. See fbstab_algorithm.h for
   * a list of adjustable options.
   * @param[in] option New option struct
   */
  void UpdateOptions(const Options& options);

  /** Returns default settings, recommended for most problems. */
  static Options DefaultOptions();
  /** Settings for increased reliability for use on hard problems. */
  static Options ReliableOptions();

 private:
  int N_ = 0;   // horizon length
  int nx_ = 0;  // number of states
  int nu_ = 0;  // number of controls
  int nc_ = 0;  // constraints per stage
  int nz_ = 0;  // number of primal variables
  int nl_ = 0;  // number of equality duals
  int nv_ = 0;  // number of inequality duals
  Options opts_;

  std::unique_ptr<Algorithm> algorithm_;
  std::unique_ptr<FullVariable> x1_;
  std::unique_ptr<FullVariable> x2_;
  std::unique_ptr<FullVariable> x3_;
  std::unique_ptr<FullVariable> x4_;
  std::unique_ptr<FullResidual> r1_;
  std::unique_ptr<FullResidual> r2_;
  std::unique_ptr<RiccatiLinearSolver> linear_solver_;
  std::unique_ptr<FullFeasibility> feasibility_checker_;

  template <class InputVariable>
  void ValidateInputSizes(const MpcData& data, const InputVariable& x) {
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
};

}  // namespace fbstab
