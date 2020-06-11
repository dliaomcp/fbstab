#pragma once

#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

#include "tools/copyable_macros.h"

namespace fbstab {

/**
 * This class is used an an input for FBstabMpc and represents a sequence of
 * matrices.
 *
 * This class owns its own memory. The assignment and copy constructor methods
 * for this class implement a deep copy.
 */
class MatrixSequence {
 public:
  FBSTAB_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(MatrixSequence)
  MatrixSequence() = default;

  /**
   * Allocates memory for (but does not initialize) a sequence.
   *
   * @param[in] len sequence length
   * @param[in] nrows number of rows
   * @param[in] ncols number of columns
   */
  MatrixSequence(int len, int nrows, int ncols = 1) {
    if (len < 0) {
      throw std::runtime_error("Negative length input in MatrixSequence");
    }
    if (nrows <= 0 || ncols <= 0) {
      throw std::runtime_error(
          "Non-positive row or column count in MatrixSequence");
    }
    N_ = len;
    nr_ = nrows;
    nc_ = ncols;
    nel_ = N_ * nr_ * nc_;
    data_.resize(nel_);
  }

  /** Returns an Eigen::Map referencing the kth Matrix in the sequence */
  Eigen::Map<Eigen::MatrixXd> operator()(int k) {
    if (k < 0 || k >= N_) {
      throw std::out_of_range("Bad indexing in MatrixSequence");
    }
    return Eigen::Map<Eigen::MatrixXd>(data() + k * nr_ * nc_, nr_, nc_);
  }

  /** Returns an Eigen::Map referencing the kth Matrix in the sequence */
  Eigen::Map<const Eigen::MatrixXd> operator()(int k) const {
    if (k < 0 || k >= N_) {
      throw std::out_of_range("Bad indexing in MatrixSequence");
    }
    return Eigen::Map<const Eigen::MatrixXd>(data() + k * nr_ * nc_, nr_, nc_);
  }

  int rows() const { return nr_; }
  int cols() const { return nc_; }
  int length() const { return N_; }
  int size() const { return nel_; }
  double* data() { return data_.data(); }
  const double* data() const { return data_.data(); }

 private:
  int N_ = 0;
  int nr_ = 1;
  int nc_ = 1;
  int nel_ = 0;
  std::vector<double> data_;
};

/**
 * This class is used an an input for FBstabMpc and represents a sequence of
 * matrices. Its purpose is to interpret existing data as a
 * sequence of matrices.
 *
 * The class assumes the data is in linear "column major" storage, i.e., if A
 * = *this and A(i,j,k) denotes the ith row and jth column, of the kth matrix in
 * the sequence then A(i,j,k) = data[k* nrows*ncols + j*nrows + i]
 *
 * The assignment and copy constructor methods for this class implement a
 * shallow copy.
 */
class MapMatrixSequence {
 public:
  FBSTAB_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(MapMatrixSequence)
  MapMatrixSequence() = default;

  /**
   * Maps a sequence onto existing memory. It is the responsibility of the user
   * to ensure the memory remains valid throughout.
   *
   * @param[in] data pointer to memory for data storage. This must be a
   * contiguous block of size len * nrows * ncols
   * @param[in] len number of matrices in the sequence > 0
   * @param[in] nrows number of rows in each matrix > 0
   * @param[in] ncols number of columns in each matrix > 0
   */
  MapMatrixSequence(const double* data, int len, int nrows, int ncols)
      : data_(data) {
    if (len <= 0) {
      throw std::runtime_error(
          "Non-positive length input in MapMatrixSequence");
    }
    if (nrows <= 0 || ncols <= 0) {
      throw std::runtime_error(
          "Non-positive row or column count in MapMatrixSequence");
    }
    if (data == nullptr) {
      throw std::runtime_error(
          "Cannot initialize MapMatrixSequence will a nullptr");
    }
    N_ = len;
    nr_ = nrows;
    nc_ = ncols;
    nel_ = N_ * nr_ * nc_;
  }

  /**
   * Creates a Map that references an MatrixSequence object. It is the
   * responsibility of the user to ensure the referenced object remains valid.
   *
   * @param[in] A object to be referenced
   */
  MapMatrixSequence(const MatrixSequence& A) : data_(A.data()) {
    N_ = A.length();
    nr_ = A.rows();
    nc_ = A.cols();
    nel_ = N_ * nr_ * nc_;
  }

  /**
   * Returns an Eigen::Map referencing the kth Matrix in the sequence
   * @param[in] k index number
   */
  Eigen::Map<const Eigen::MatrixXd> operator()(int k) const {
    if (k < 0 || k >= N_) {
      throw std::out_of_range("Bad indexing in MapMatrixSequence");
    }
    if (data_ == nullptr) {
      throw std::runtime_error(
          "In MapMatrixSequence, cannot index into null data.");
    }
    return Eigen::Map<const Eigen::MatrixXd>(data_ + k * nr_ * nc_, nr_, nc_);
  }

  int rows() const { return nr_; }
  int cols() const { return nc_; }
  int length() const { return N_; }
  int size() const { return nel_; }
  const double* data() { return data_; }
  const double* data() const { return data_; }

 private:
  const double* data_ = nullptr;
  int N_ = 0;
  int nr_ = 1;
  int nc_ = 1;
  int nel_ = 0;
};

}  // namespace fbstab