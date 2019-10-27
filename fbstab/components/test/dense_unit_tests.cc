/**
 * @file Runs unit tests for the Dense MPC components. See
 * dense_component_unit_tests.h for documentation.
 */
#include "fbstab/components/test/dense_unit_tests.h"

#include <gtest/gtest.h>

namespace fbstab {
namespace test {
namespace {

GTEST_TEST(FBstabDense, DenseVariable) {
  DenseComponentUnitTests test;
  test.DenseVariableTests();
}

GTEST_TEST(FBstabDense, TestInnerResidualCalculation) {
  DenseComponentUnitTests test;
  test.InnerResidualCalculation();
}

GTEST_TEST(FBstabDense, NaturalResidualCalculation) {
  DenseComponentUnitTests test;
  test.NaturalResidualCalculation();
}

GTEST_TEST(FBstabDense, DenseLinearSolver) {
  DenseComponentUnitTests test;
  test.LinearSolverResidual();
}

GTEST_TEST(FBstabDense, PrimalInfeasibilityDetection) {
  DenseComponentUnitTests test;
  test.PrimalInfeasibilityDetection();
}

GTEST_TEST(FBstabDense, DualInfeasibilityDetection) {
  DenseComponentUnitTests test;
  test.DualInfeasibilityDetection();
}

}  // namespace
}  // namespace test
}  // namespace fbstab
