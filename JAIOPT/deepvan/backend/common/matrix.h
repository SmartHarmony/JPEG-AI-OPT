#ifndef DEEPVAN_BACKEND_COMMON_MATRIX_H_
#define DEEPVAN_BACKEND_COMMON_MATRIX_H_

#include "deepvan/core/types.h"
#include "deepvan/utils/logging.h"

namespace deepvan {

enum MatrixMajor {
  RowMajor,
  ColMajor
};

inline MatrixMajor TransposeMatrixMajor(const MatrixMajor src_major) {
  return src_major == RowMajor ? ColMajor : RowMajor;
}

template<typename T>
class MatrixMap {
 public:
  MatrixMap()
      : data_(nullptr),
        matrix_major_(RowMajor),
        rows_(0),
        cols_(0),
        stride_(0) {}
  MatrixMap(T *data,
            const MatrixMajor matrix_major,
            const index_t rows,
            const index_t cols) :
      data_(data),
      matrix_major_(matrix_major),
      rows_(rows),
      cols_(cols),
      stride_(matrix_major == ColMajor ? rows : cols) {}
  MatrixMap(T *data,
            const MatrixMajor matrix_major,
            const index_t rows,
            const index_t cols,
            const index_t stride) :
      data_(data),
      matrix_major_(matrix_major),
      rows_(rows),
      cols_(cols),
      stride_(stride) {}
  MatrixMap(const MatrixMap &other)
      : data_(other.data_),
        matrix_major_(other.matrix_major_),
        rows_(other.rows_),
        cols_(other.cols_),
        stride_(other.stride_) {}

  MatrixMajor matrix_major() const { return matrix_major_; }
  index_t rows() const { return rows_; }
  index_t cols() const { return cols_; }
  index_t stride() const { return stride_; }
  int rows_stride() const {
    return matrix_major_ == MatrixMajor::ColMajor ? 1 : stride_;
  }
  int cols_stride() const {
    return matrix_major_ == MatrixMajor::RowMajor ? 1 : stride_;
  }
  index_t size() const { return rows_ * cols_; }
  T *data() const { return data_; }
  T *data(int rows, int cols) const {
    return data_ + rows * rows_stride() + cols * cols_stride();
  }
  T &operator()(int row, int col) const { return *data(row, col); }
  MatrixMap block(int start_row, int start_col, int block_rows,
                  int block_cols) const {
    CONDITIONS(start_row >= 0, "Error");
    CONDITIONS(start_row + block_rows <= rows_, "Error");
    CONDITIONS(start_col >= 0, "Error");
    CONDITIONS(start_col + block_cols <= cols_, "Error");

    return MatrixMap(data(start_row, start_col),
                     matrix_major_,
                     block_rows,
                     block_cols,
                     stride_);
  }

 private:
  T *data_;
  MatrixMajor matrix_major_;
  index_t rows_;
  index_t cols_;
  index_t stride_;
};

}  // namespace deepvan

#endif  // DEEPVAN_BACKEND_COMMON_MATRIX_H_
