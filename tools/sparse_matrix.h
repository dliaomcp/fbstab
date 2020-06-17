#pragma once

namespace fbstab {

/**
 * A glorified structure for
 */
class CompressedColumnMatrix {
 public:
  CompressedColumnMatrix(nrows, ncols, nnz) {
    idx_outer_ = tools::make_unique<int[]>(ncols + 1);
    idx_inner_ = tools::make_unique<int[]>(nnz);
    data_ = tools::make_unique<double[]>(nnz);
  }
  double* x() { return data_.get(); }
  int* i() { return idx_inner_.get(); }
  int* p() { return idx_outer_.get(); }

  int nnz() { return nnz_; }
  int rows() { return nr_; }
  int cols() { return nc_; }

 private:
  nr_ = 0;
  nc_ = 0;
  nnz_ = 0;
  std::unique_ptr<int[]> idx_outer_;
  std::unique_ptr<int[]> idx_inner_;
  std::unique_ptr<double[]> data_;
};

}  // namespace fbstab