#include <gtest/gtest.h>

#include <Eigen/Dense>

namespace fbstab {
namespace test {

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

GTEST_TEST(A, B) {
  int n = 3;
  int l = 4;

  MatrixXd H = Eigen::MatrixXd::Identity(n, n);
  MatrixXd G = Eigen::MatrixXd::Random(l, n);

  MatrixXd K(n + l, n + l);

  K.block(0, 0, n, n) = H;
  K.block(n, 0, l, n) = G;
  K.block(n, n, l, l) = -Eigen::MatrixXd::Identity(l, l);

  std::cout << K << std::endl;

  Eigen::LDLT<Eigen::MatrixXd> ldlt;
  ldlt.compute(K);

  MatrixXd L = ldlt.matrixL();
  std::cout << L << std::endl;

  VectorXd d = ldlt.vectorD();
  std::cout << d << std::endl;
}

GTEST_TEST(A, C) {
  int n = 3;
  int l = 0;

  MatrixXd H = Eigen::MatrixXd::Identity(n, n);
  MatrixXd G = Eigen::MatrixXd::Random(l, n);

  MatrixXd K(n + l, n + l);

  K.block(0, 0, n, n) = H;
  K.block(n, 0, l, n) = G;
  K.block(n, n, l, l) = -Eigen::MatrixXd::Identity(l, l);

  std::cout << K << std::endl;

  Eigen::LDLT<Eigen::MatrixXd> ldlt;
  ldlt.compute(K);

  MatrixXd L = ldlt.matrixL();
  std::cout << L << std::endl;

  VectorXd d = ldlt.vectorD();
  std::cout << d << std::endl;
}

GTEST_TEST(A, D) {
  int nr = 4;
  int nc = 1;

  MatrixXd H(nr, nc);
  H << 1, 2, 3, 4;

  Eigen::Map<Eigen::MatrixXd> A(H.data(), nr, nc);

  Eigen::VectorXd a = A;

  std::cout << A << std::endl;
  std::cout << a << std::endl;

  const auto& AA = A;

  std::cout << AA << std::endl;
}

}  // namespace test
}  // namespace fbstab