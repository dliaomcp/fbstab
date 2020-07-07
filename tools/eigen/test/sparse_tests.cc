#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace fbstab {
namespace test {

using SparseMatrix = Eigen::SparseMatrix<double>;
using SparseMap = Eigen::Map<SparseMatrix>;
GTEST_TEST(EigenSparse, AssignDiagonal) {
  int n = 4;
  SparseMatrix A(n, n);
  A.setIdentity();

  Eigen::VectorXd d(n);
  d << 1, 2, 3, 4;
  A.diagonal() += d;

  std::cout << A << std::endl;
}

GTEST_TEST(EigenSparse, Concatenate) {}

GTEST_TEST(EigenSparse, BuildShurTest) {
  int n = 4;
  int v = 5;
  SparseMatrix E(4, 4);
  E.setIdentity();

  Eigen::MatrixXd Ad = Eigen::MatrixXd::Random(v, n);
  SparseMatrix A = Ad.sparseView();

  E += (A.transpose() * A).triangularView<Eigen::Upper>();

  std::cout << E << std::endl;
}

GTEST_TEST(EigenSparse, BuildShurComplement) {}

}  // namespace test
}  // namespace fbstab