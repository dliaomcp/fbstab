#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "tools/qdldl/qdldl_wrapper.h"
#include "tools/utilities.h"

namespace fbstab {
namespace tools {
namespace test {

using SparseMatrix = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

GTEST_TEST(QDLDL, SparseMatrixTest) {
  int nrows = 4;
  int ncols = 6;

  int icol[] = {0, 2, 3, 5, 8, 9, 13};
  int irow[] = {0, 3, 1, 2, 3, 0, 2, 3, 1, 0, 1, 2, 3};
  double vals[] = {11, 41, 22, 33, 43, 14, 34, 44, 25, 16, 26, 36, 46};
  const int nnz = sizeof(vals) / sizeof(vals[0]);

  Eigen::Map<SparseMatrix> A(nrows, ncols, nnz, &icol[0], &irow[0], &vals[0],
                             nullptr);

  std::cout << A << std::endl;
}

// A modified version of
// https://github.com/oxfordcontrol/qdldl/blob/master/examples/example.c
// using Eigen sparse matrices
GTEST_TEST(QDLDL, FactorizationTest) {
  const int n = 10;
  int Ap[] = {0, 1, 2, 4, 5, 6, 8, 10, 12, 14, 17};
  int Ai[] = {0, 1, 1, 2, 3, 4, 1, 5, 0, 6, 3, 7, 6, 8, 1, 2, 9};
  double Ax[] = {1.0,        0.460641,   -0.121189, 0.417928,  0.177828,
                 0.1,        -0.0290058, -1.0,      0.350321,  -0.441092,
                 -0.0845395, -0.316228,  0.178663,  -0.299077, 0.182452,
                 -1.56506,   -0.1};

  // Map the data into an eigen type
  const int nnz = sizeof(Ax) / sizeof(Ax[0]);
  Eigen::Map<SparseMatrix> B(n, n, nnz, &Ap[0], &Ai[0], &Ax[0], nullptr);
  SparseMatrix A = B;
  Eigen::VectorXd b(n);
  b << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
  Eigen::VectorXd x = b;

  QdldlWrapper LDL(n, Ap, Ai);
  LDL.Factor(A);
  LDL.Solve(&x);

  Eigen::VectorXd r;
  r.noalias() = A.selfadjointView<Eigen::Upper>() * x - b;
  ASSERT_LE(r.norm(), 1e-12);
}

}  // namespace test
}  // namespace tools
}  // namespace fbstab