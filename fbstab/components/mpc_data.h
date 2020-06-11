#pragma once

#include <Eigen/Dense>
#include <vector>

#include "fbstab/components/abstract_components.h"
#include "tools/copyable_macros.h"
#include "tools/matrix_sequence.h"

namespace fbstab {

// Forward declaration of testing class to enable a friend declaration.
namespace test {
class MpcComponentUnitTests;
}  // namespace test

/**
 * This class represents data for quadratic programming problems of the
 * following type (1):
 *
 *     min.  \sum_{i=0}^N 1/2 [x(i)]' * [Q(i) S(i)'] [x(i)] + [q(i)]'*[x(i)]
 *                            [u(i)]    [S(i) R(i) ] [u(i)]   [r(i)]  [u(i)]
 *     s.t.  x(i+1) = A(i)*x(i) + B(i) u(i) + c(i), i = 0 ... N-1
 *           x(0) = x0,
 *           E(i)*x(i) + L(i)*u(i) + d(i) <= 0,     i = 0 ... N
 *
 * The horizon length is N, the dimension of x(i) is nx, of u(i) is nu,
 * and the number of constraints per stage is nc.
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
 * This class contains storage and methods for implicitly working with the
 * compact representation (2) in an efficient manner.
 */
class MpcData : public Data {
 public:
  FBSTAB_NO_COPY_NO_MOVE_NO_ASSIGN(MpcData)
  /**
   * Creates problem data and performs input validation. Throws
   * a runtime_error if the problem data aren't consistently sized.
   *
   * This class doesn't take ownership of the input data and assumes all input
   * pointers remain valid.
   *
   * All arguments are inputs and point to data defining a linear-quadratic
   * optimal control problem, see the class comment.
   *
   * The template allows MatrixSequences/Eigen::VectorXds and
   * MapMatrixSequences/Eigen::Map<Eigen::VectorXd> input combinations.
   *
   * @tparam Sequence: MatrixSequence or MapMatrixSequence
   * @tparam Vector: Eigen::VectorXd or Eigen::Map<Eigen::VectorXd>
   */
  template <class Sequence, class Vector>
  MpcData(const Sequence* Q, const Sequence* R, const Sequence* S,
          const Sequence* q, const Sequence* r, const Sequence* A,
          const Sequence* B, const Sequence* c, const Sequence* E,
          const Sequence* L, const Sequence* d, const Vector* x0)
      : Q_(*Q),
        R_(*R),
        S_(*S),
        q_(*q),
        r_(*r),
        A_(*A),
        B_(*B),
        c_(*c),
        E_(*E),
        L_(*L),
        d_(*d),
        x0_(x0->data(), x0->size()) {
    ValidateInputs();

    N_ = B_.length();
    nx_ = B_.rows();
    nu_ = B_.cols();
    nc_ = E_.rows();
    nz_ = (N_ + 1) * (nx_ + nu_);
    nl_ = (N_ + 1) * nx_;
    nv_ = (N_ + 1) * nc_;

    // Compute the forcing norm = ||[f,h,b]||
    forcing_norm_ = 0.0;
    for (int i = 0; i < N_ + 1; i++) {
      forcing_norm_ += q_(i).squaredNorm();
      forcing_norm_ += r_(i).squaredNorm();
      forcing_norm_ += d_(i).squaredNorm();
      forcing_norm_ += (i == 0 ? x0_.squaredNorm() : c_(i - 1).squaredNorm());
    }
    forcing_norm_ = sqrt(forcing_norm_);
  }
  /**
   * Computes the operation y <- a*H*x + b*y without forming H explicitly.
   * This implements a BLAS operation, see
   * http://www.netlib.org/blas/blasqr.pdf.
   *
   * @param[in] x Input vector, length(x) = (nx+nu)*(N+1)
   * @param[in] a Input scaling
   * @param[in] b Scaling
   * @param[in,out] y Output vector, length(y) = (nx+nu)*(N+1)
   *
   * Throws a runtime_error if sizes aren't consistent or y is null.
   */
  void gemvH(const Eigen::VectorXd& x, double a, double b,
             Eigen::VectorXd* y) const;

  /**
   * Computes y <- a*A*x + b*y without forming A explicitly.
   * This implements a BLAS operation, see
   * http://www.netlib.org/blas/blasqr.pdf.
   * @param[in] x Input vector, length(x) = (nx+nu)*(N+1)
   * @param[in] a Input scaling
   * @param[in] b Scaling
   * @param[in,out] y Output vector, length(y) = nc*(N+1)
   *
   * Throws a runtime_error if sizes aren't consistent or y is null.
   */
  void gemvA(const Eigen::VectorXd& x, double a, double b,
             Eigen::VectorXd* y) const;

  /**
   * Computes y <- a*G*x + b*y without forming G explicitly
   * This implements a BLAS operation, see
   * http://www.netlib.org/blas/blasqr.pdf.
   * @param[in] x Input vector, length(x) = (nx+nu)*(N+1)
   * @param[in] a Input scaling
   * @param[in] b Scaling
   * @param[in,out] y Output vector, length(y) = nx*(N+1)
   *
   * Throws a runtime_error if sizes aren't consistent or y is null.
   */
  void gemvG(const Eigen::VectorXd& x, double a, double b,
             Eigen::VectorXd* y) const;

  /**
   * Computes y <- a*A'*x + b*y without forming A explicitly
   * This implements a BLAS operation, see
   * http://www.netlib.org/blas/blasqr.pdf.
   * @param[in] x Input vector, length(x) = nc*(N+1)
   * @param[in] a Input scaling
   * @param[in] b Scaling
   * @param[in,out] y Output vector, length(y) = (nx+nu)*(N+1)
   *
   * Throws a runtime_error if sizes aren't consistent or y is null.
   */
  void gemvAT(const Eigen::VectorXd& x, double a, double b,
              Eigen::VectorXd* y) const;

  /**
   * Computes y <- a*G'*x + b*y without forming G explicitly
   * This implements a BLAS operation, see
   * http://www.netlib.org/blas/blasqr.pdf.
   * @param[in] x Input vector, length(x) = (nx)*(N+1)
   * @param[in] a Input scaling
   * @param[in] b Scaling
   * @param[in,out] y Output vector, length(y) = (nx+nu)*(N+1)
   *
   * Throws a runtime_error if sizes aren't consistent or y is null.
   */
  void gemvGT(const Eigen::VectorXd& x, double a, double b,
              Eigen::VectorXd* y) const;

  /**
   * Computes y <- a*f + y without forming f explicitly.
   * This implements a BLAS operation, see
   * http://www.netlib.org/blas/blasqr.pdf.
   * @param[in] a Scaling factor
   * @param[in,out] y Output vector, length(y) = (nx+nu)*(N+1)
   *
   * Throws a runtime_error if sizes aren't consistent or y is null.
   */
  void axpyf(double a, Eigen::VectorXd* y) const;

  /**
   * Computes y <- a*h + y without forming h explicitly.
   * This implements a BLAS operation, see
   * http://www.netlib.org/blas/blasqr.pdf.
   * @param[in] a Scaling factor
   * @param[in,out] y Output vector, length(y) = nx*(N+1)
   *
   * Throws a runtime_error if sizes aren't consistent or y is null.
   */
  void axpyh(double a, Eigen::VectorXd* y) const;

  /**
   * Computes y <- a*b + y without forming b explicitly.
   * This implements a BLAS operation, see
   * http://www.netlib.org/blas/blasqr.pdf.
   * @param[in] a Scaling factor
   * @param[in,out] y Output vector, length(y) = nc*(N+1)
   *
   * Throws a runtime_error if sizes aren't consistent or y is null.
   */
  void axpyb(double a, Eigen::VectorXd* y) const;

  double ForcingNorm() const { return forcing_norm_; }

  int N() const { return N_; }
  int nx() const { return nx_; }
  int nu() const { return nu_; }
  int nc() const { return nc_; }

 private:
  int N_ = 0;   // horizon length
  int nx_ = 0;  // number of states
  int nu_ = 0;  // number of controls
  int nc_ = 0;  // constraints per stage
  int nz_ = 0;  // number of primal variables
  int nl_ = 0;  // number of equality duals
  int nv_ = 0;  // number of inequality duals

  double forcing_norm_ = 0.0;

  const MapMatrixSequence Q_;
  const MapMatrixSequence R_;
  const MapMatrixSequence S_;
  const MapMatrixSequence q_;
  const MapMatrixSequence r_;
  const MapMatrixSequence A_;
  const MapMatrixSequence B_;
  const MapMatrixSequence c_;
  const MapMatrixSequence E_;
  const MapMatrixSequence L_;
  const MapMatrixSequence d_;
  const Eigen::Map<const Eigen::VectorXd> x0_;

  // Throws an exception if any of the inputs have inconsistent sizes.
  void ValidateInputs() const;

  friend class test::MpcComponentUnitTests;
  friend class RiccatiLinearSolver;
  friend class FBstabMpc;
};

}  // namespace fbstab
