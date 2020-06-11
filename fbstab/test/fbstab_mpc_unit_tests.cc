#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "fbstab/fbstab_mpc.h"
#include "fbstab/test/ocp_generator.h"

namespace fbstab {
namespace test {

using VectorXd = Eigen::VectorXd;

GTEST_TEST(FBstabMpc, DoubleIntegrator) {
  // Get the problem data.
  OcpGenerator ocp;
  ocp.DoubleIntegrator(2);  // horizon length of 2
  FBstabMpc::ProblemData data = ocp.GetFBstabInput();

  FBstabMpc::Variable x(ocp.ProblemSize());
  FBstabMpc solver(ocp.ProblemSize());

  FBstabMpc::Options opts = FBstabMpc::DefaultOptions();
  opts.abs_tol = 1e-8;
  opts.display_level = Display::OFF;
  solver.UpdateOptions(opts);
  SolverOut out = solver.Solve(data, &x);

  ASSERT_EQ(out.eflag, ExitFlag::SUCCESS);
  ASSERT_LE(out.residual, 1e-6);

  VectorXd zopt(ocp.nz());
  VectorXd lopt(ocp.nl());
  VectorXd vopt(ocp.nv());

  // Computed using MATLAB's quadprog command.
  zopt << -5.31028204670497e-14, 5.02854354118183e-13, 0.311688311338095,
      5.35637944798588e-13, 0.311688311339015, -0.0779220779990502,
      0.311688311339667, 0.233766233340057, -0.103896103779874;

  lopt << -5.24675324688535, -4.49350649223710, -3.55844155822323,
      -0.935064934014372, -1.48051948022526, 0.233766233996585;

  vopt << 1.06213597221667e-13, -1.41190425869539e-21, 0, 0, 0, 0,
      -1.50393600622818e-21, -8.75144622575045e-10, 0, 0, 0, 0,
      -8.75144611157041e-10, -6.56358459377444e-10, 0, 0, 0, 0;

  for (int i = 0; i < ocp.nz(); i++) {
    EXPECT_NEAR(x.z(i), zopt(i), 1e-8);
  }

  for (int i = 0; i < ocp.nl(); i++) {
    EXPECT_NEAR(x.l(i), lopt(i), 1e-8);
  }

  for (int i = 0; i < ocp.nv(); i++) {
    EXPECT_NEAR(x.v(i), vopt(i), 1e-8);
  }
}

GTEST_TEST(FBstabMpc, DoubleIntegratorLongHorizon) {
  // Get the problem data.
  OcpGenerator ocp;
  ocp.DoubleIntegrator(20);  // horizon length of 20
  FBstabMpc::ProblemDataRef data = ocp.GetFBstabInputRef();

  // Set up the initial guess.
  FBstabMpc::Variable x(ocp.ProblemSize());

  // Call the solver.
  FBstabMpc solver(ocp.ProblemSize());

  FBstabMpc::Options opts = FBstabMpc::DefaultOptions();
  opts.abs_tol = 1e-8;
  opts.display_level = Display::OFF;
  solver.UpdateOptions(opts);
  SolverOut out = solver.Solve(data, &x);

  ASSERT_EQ(out.eflag, ExitFlag::SUCCESS);
  ASSERT_LE(out.residual, 1e-6);
}

GTEST_TEST(FBstabMpc, ServoMotor) {
  // Get the problem data.
  OcpGenerator ocp;
  ocp.ServoMotor(25);  // horizon length of 25
  FBstabMpc::ProblemData data = ocp.GetFBstabInput();

  // Set up the initial guess.
  FBstabMpc::Variable x(ocp.ProblemSize());

  // Call the solver.
  FBstabMpc solver(ocp.ProblemSize());

  FBstabMpc::Options opts = FBstabMpc::DefaultOptions();
  opts.abs_tol = 1e-8;
  opts.display_level = Display::OFF;
  solver.UpdateOptions(opts);
  SolverOut out = solver.Solve(data, &x);

  ASSERT_EQ(out.eflag, ExitFlag::SUCCESS);
  ASSERT_LE(out.residual, 1e-6);
}

GTEST_TEST(FBstabMpc, SpacecraftRelativeMotion) {
  // Get the problem data.
  OcpGenerator ocp;
  ocp.SpacecraftRelativeMotion(40);  // horizon length of 40
  FBstabMpc::ProblemData data = ocp.GetFBstabInput();

  // Set up the initial guess.
  FBstabMpc::Variable x(ocp.ProblemSize());

  // Call the solver.
  FBstabMpc solver(ocp.ProblemSize());

  FBstabMpc::Options opts = FBstabMpc::DefaultOptions();
  opts.abs_tol = 1e-8;
  opts.display_level = Display::OFF;
  solver.UpdateOptions(opts);
  SolverOut out = solver.Solve(data, &x);

  ASSERT_EQ(out.eflag, ExitFlag::SUCCESS);
  ASSERT_LE(out.residual, 1e-6);
}

GTEST_TEST(FBstabMpc, CopolymerizationReactor) {
  // Get the problem data.
  OcpGenerator ocp;
  ocp.CopolymerizationReactor(80);  // horizon length of 80
  FBstabMpc::ProblemData data = ocp.GetFBstabInput();

  // Set up the initial guess.
  FBstabMpc::Variable x(ocp.ProblemSize());

  // Call the solver.
  FBstabMpc solver(ocp.ProblemSize());

  FBstabMpc::Options opts = FBstabMpc::DefaultOptions();
  opts.abs_tol = 1e-8;
  opts.display_level = Display::OFF;
  solver.UpdateOptions(opts);
  SolverOut out = solver.Solve(data, &x);

  ASSERT_EQ(out.eflag, ExitFlag::SUCCESS);
  ASSERT_LE(out.residual, 1e-6);
}

}  // namespace test
}  // namespace fbstab
