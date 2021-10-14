#pragma once

#include <Eigen/Dense>

#include "qdldl/include/qdldl.h"

namespace fbstab {

/**
 * A wrapper class providing a streamlined interface to the QDLDL package.
 *
 * This class computes the factorization
 *
 *     A = LDL'
 *
 * where A is symmetric quasi-definite and stored in upper triangular compressed
 * column format.
 */
class QdldlWrapper {
 public:
  // Analyzes the sparsity pattern given by Ap and Ai and prepares (elimination
  // tree computation, memory allocation) to factor matrices with that
  // structure.
  QdldlWrapper(int n, const int* Ap, const int* Ai) {
    n_ = n;

    // Allocate working memory
    iwork_.resize(3 * n_);
    bwork_.resize(n_);
    fwork_.resize(n_);

    // Memory for the elimination tree outputs
    etree_.resize(n_);
    Lnz_.resize(n_);

    // Perform the factorization sparsity analysis
    nnz_ = QDLDL_etree(n_, Ap, Ai, iwork_.data(), Lnz_.data(), etree_.data());

    // Allocate memory for the LDL factors
    Lp_.resize(n_ + 1);
    Li_.resize(nnz_);
    Lx_.resize(nnz_);
    D_.resize(n_);
    Dinv_.resize(n_);
  }

  template <class SparseMatrix>
  void Factor(const SparseMatrix& A) {
    // TODO: check that A is the same size as was declared in the constructor
    // TODO: check that A is in compressed form
    QDLDL_factor(n_, A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr(),
                 Lp_.data(), Li_.data(), Lx_.data(), D_.data(), Dinv_.data(),
                 Lnz_.data(), etree_.data(), bwork_.data(), iwork_.data(),
                 fwork_.data());
  }

  // This solve is performed inplace.
  void Solve(Eigen::VectorXd* x) {
    QDLDL_solve(n_, Lp_.data(), Li_.data(), Lx_.data(), Dinv_.data(),
                x->data());
  }

 private:
  int nnz_ = 0;  // number of non-zeros in L
  int n_ = 0;    // size of A

  // Elimination tree
  std::vector<int> etree_;
  std::vector<int> Lnz_;

  // The lower triangular factor in CCS form
  std::vector<int> Li_;     // inner indices
  std::vector<int> Lp_;     // outer indices
  std::vector<double> Lx_;  // values

  // Diagonal matrix
  std::vector<double> D_;
  std::vector<double> Dinv_;

  // Working memory
  std::vector<int> iwork_;
  std::vector<double> fwork_;
  std::vector<unsigned char> bwork_;
};

}  // namespace fbstab