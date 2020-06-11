#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>

#include "fbstab/fbstab_dense.h"
#include "tools/utilities.h"

namespace fbstab {
namespace test {

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;
/**
 * Tests FBstab with
 *
 * H = [3 1]  f = [10]
 *     [1 1]      [5 ]
 *
 * A = [-1 0] b = [0]
 *     [0  1]     [0]
 *
 * This QP can be solved analytically
 * and has the unique primal(z) - dual(v) solution
 * z = [0 -5],  v = [5 0]
 */
GTEST_TEST(FBstabDense, FeasibleQP) {
  int n = 2;
  int m = 0;
  int q = 2;

  FBstabDense::Variable x0(n, m, q);
  FBstabDense::ProblemData data(n, m, q);
  data.H << 3, 1, 1, 1;
  data.f << 10, 5;
  data.A << -1, 0, 0, 1;
  data.b << 0, 0;

  FBstabDense solver(n, m, q);
  FBstabDense::Options opts = FBstabDense::DefaultOptions();
  opts.abs_tol = 1e-8;
  opts.display_level = Display::OFF;
  solver.UpdateOptions(opts);

  SolverOut out = solver.Solve(data, &x0);

  ASSERT_EQ(out.eflag, ExitFlag::SUCCESS);

  VectorXd zopt(2);
  VectorXd vopt(2);
  zopt << 0, -5;
  vopt << 5, 0;
  for (int i = 0; i < n; i++) {
    EXPECT_NEAR(x0.z(i), zopt(i), 1e-8);
  }

  for (int i = 0; i < q; i++) {
    EXPECT_NEAR(x0.v(i), vopt(i), 1e-8);
  }
}

/**
 * Tests FBstab with
 *
 * H = [4 1]  f = [1]
 *     [1 2]      [1]
 *
 * G = [1 1]  h = [1]
 *
 * A = [-1 0] b = [0]
 *     [0 -1]     [0]
 *
 */
GTEST_TEST(FBstabDense, FeasibleQPwithEQ) {
  int n = 2;
  int m = 1;
  int q = 2;

  FBstabDense::Variable x0(n, m, q);
  FBstabDense::ProblemData data(n, m, q);
  data.H << 4, 1, 1, 2;
  data.f << 1, 1;
  data.G << 1, 1;
  data.h << 1;
  data.A << -1, 0, 0, -1;
  data.b << 0, 0;

  FBstabDense solver(n, m, q);
  FBstabDense::Options opts = FBstabDense::DefaultOptions();
  opts.abs_tol = 1e-8;
  opts.display_level = Display::OFF;
  solver.UpdateOptions(opts);

  SolverOut out = solver.Solve(data, &x0);

  ASSERT_EQ(out.eflag, ExitFlag::SUCCESS);

  VectorXd zopt(n);
  zopt << 2.5e-1, 7.5e-1;
  for (int i = 0; i < n; i++) {
    EXPECT_NEAR(x0.z(i), zopt(i), 1e-8);
  }
}

/**
 * Tests FBstab with
 *
 * H = [1 0]  f = [1]
 *     [0 0]      [0]
 *
 * A = [0  0] b = [0 ]
 *     [1  0]     [3 ]
 *     [0  1]     [3 ]
 *     [-1 0]     [-1]
 *     [0 -1]     [-1]
 *
 * This QP is degenerate with a primal solution set
 * [1] x [1,3]
 */
GTEST_TEST(FBstabDense, DegenerateQP) {
  constexpr int n = 2;
  constexpr int m = 0;
  constexpr int q = 5;

  // Using a ref variable
  std::unique_ptr<double[]> zmem(new double[n]);
  std::unique_ptr<double[]> lmem(new double[m]);
  std::unique_ptr<double[]> vmem(new double[q]);
  std::unique_ptr<double[]> ymem(new double[q]);

  Eigen::Map<VectorXd> z(zmem.get(), n);
  Eigen::Map<VectorXd> l(lmem.get(), m);
  Eigen::Map<VectorXd> v(vmem.get(), q);
  Eigen::Map<VectorXd> y(ymem.get(), q);

  FBstabDense::VariableRef x0(&z, &l, &v, &y);
  x0.fill(0.0);

  std::unique_ptr<double[]> Hmem(new double[n * n]);
  std::unique_ptr<double[]> fmem(new double[n]);
  std::unique_ptr<double[]> Gmem(new double[m * n]);
  std::unique_ptr<double[]> hmem(new double[m]);
  std::unique_ptr<double[]> Amem(new double[q * n]);
  std::unique_ptr<double[]> bmem(new double[q]);

  Eigen::Map<MatrixXd> H(Hmem.get(), n, n);
  Eigen::Map<VectorXd> f(fmem.get(), n);
  Eigen::Map<MatrixXd> A(Amem.get(), q, n);
  Eigen::Map<VectorXd> b(bmem.get(), q);
  Eigen::Map<MatrixXd> G(Gmem.get(), m, n);
  Eigen::Map<VectorXd> h(hmem.get(), m);
  H << 1, 0, 0, 0;
  f << 1, 0;
  A << 0, 0, 1, 0, 0, 1, -1, 0, 0, -1;
  b << 0, 3, 3, -1, -1;

  FBstabDense::ProblemDataRef data(&H, &f, &G, &h, &A, &b);

  FBstabDense solver(n, m, q);
  FBstabDense::Options opts = FBstabDense::DefaultOptions();
  opts.abs_tol = 1e-8;
  opts.display_level = Display::OFF;
  solver.UpdateOptions(opts);

  SolverOut out = solver.Solve(data, &x0);

  ASSERT_EQ(out.eflag, ExitFlag::SUCCESS);
  EXPECT_NEAR(x0.z(0), 1, 1e-8);
  EXPECT_TRUE((x0.z(1) >= 1) && (x0.z(1) <= 3));

  // Check satisfaction of KKT conditions.
  VectorXd r1 = data.H * x0.z + data.f + data.A.transpose() * x0.v;
  VectorXd r2 = x0.y.cwiseMin(x0.v);

  ASSERT_NEAR(r1.norm() + r2.norm(), 0, 1e-6);
}

/**
 * Tests FBstab with
 *
 * H = [1 0]  f = [1 ]
 *     [0 0]      [-1]
 *
 * A = [1  1] b = [0 ]
 *     [1  0]     [3 ]
 *     [0  1]     [3 ]
 *     [-1 0]     [-1]
 *     [0 -1]     [-1]
 *
 * This QP is infeasible, i.e.,
 * there is no z satisfying Az <= b
 */

GTEST_TEST(FBstabDense, InfeasibleQP) {
  int n = 2;
  int m = 0;
  int q = 5;

  FBstabDense::ProblemData data(n, m, q);
  data.H << 1, 0, 0, 0;
  data.f << 1, -1;
  data.A << 1, 1, 1, 0, 0, 1, -1, 0, 0, -1;
  data.b << 0, 3, 3, -1, -1;

  FBstabDense::Variable x0(n, m, q);

  FBstabDense solver(n, m, q);
  FBstabDense::Options opts = FBstabDense::DefaultOptions();
  opts.abs_tol = 1e-8;
  opts.display_level = Display::OFF;
  solver.UpdateOptions(opts);

  SolverOut out = solver.Solve(data, &x0);

  ASSERT_EQ(out.eflag, ExitFlag::PRIMAL_INFEASIBLE);
}

/**
 * Tests FBstab with
 *
 * H = [1 0]  f = [1 ]
 *     [0 0]      [-1]
 *
 * A = [0  0] b = [0 ]
 *     [1  0]     [3 ]
 *     [-1 0]     [-1]
 *     [0 -1]     [-1]
 *
 * This QP is unbounded below, i.e.,
 * its optimal value is -infinity
 */
GTEST_TEST(FBstabDense, UnboundedQP) {
  int n = 2;
  int m = 0;
  int q = 4;

  FBstabDense::ProblemData data(n, m, q);
  data.H << 1, 0, 0, 0;
  data.f << 1, -1;
  data.A << 0, 0, 1, 0, -1, 0, 0, -1;
  data.b << 0, 3, -1, -1;

  FBstabDense::Variable x0(n, m, q);

  FBstabDense solver(n, m, q);

  FBstabDense::Options opts = FBstabDense::DefaultOptions();
  opts.abs_tol = 1e-8;
  opts.display_level = Display::OFF;
  solver.UpdateOptions(opts);

  SolverOut out = solver.Solve(data, &x0);

  ASSERT_EQ(out.eflag, ExitFlag::DUAL_INFEASIBLE);
}

}  // namespace test
}  // namespace fbstab
