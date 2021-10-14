#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "fbstab/components/abstract_components.h"
#include "fbstab/components/full_residual.h"
#include "fbstab/components/full_variable.h"
#include "fbstab/components/sparse_data.h"
#include "tools/copyable_macros.h"

namespace fbstab {
class SparseLDLSolver
    : public LinearSolver<FullVariable, FullResidual, SparseData> {
  using SparseMatrix = Eigen::SparseMatrix<double>;
  using SparseMap = Eigen::Map<SparseMatrix>;

 public:
  FBSTAB_NO_COPY_NO_MOVE_NO_ASSIGN(SparseLDLSolver)

  /**
   * Prepares to solve linear systems based on specified sparsity patterns.
   * This constructor:
   * - Allocates workspace memory
   * - Performs a symbolic factorization analysis
   *
   * The matrix inputs are treated as sparsity patterns.
   *
   * @param[in] H  nz x nz Hesssian
   * @param[in] G  nl x nz Equality constraint matrix (nl = 0 is OK!)
   * @param[in] A  nv x nz Inequality constraint matrix
   */
  SparseLDLSolver(const SparseMap& H, const SparseMap& G, const SparseMap& A) {
    nz_ = H.rows();
    nl_ = G.rows();
    nv_ = A.rows();
    // Check for size consistency

    // Get the sparsity pattern of E = H + S + A'*A
    SparseMatrix E(nz_, nz_);
    E.setIdentity();
    E += H.triangularView<Eigen::Upper>();
    E += (A.transpose() * A).triangularView<Eigen::Upper>();

    nnzE_ = E.nonZeros();
    // Allocate memory for G in tripplet form

    nnzG_ = G.nonZeros();

    // get everything into a tripplet form

    // Create a concatenated matrix and factor it!
  }

 private:
};

}  // namespace fbstab