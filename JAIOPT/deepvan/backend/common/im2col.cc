#include "deepvan/backend/common/im2col.h"

namespace deepvan {
namespace im2col {

bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

void get_left_right_pad_size(int pad_w, int kernel_col, int &pad_l,
                             int &pad_r) {
  int remain_pad = pad_w - kernel_col;
  pad_l = std::max(remain_pad, 0);
  pad_r = std::abs(std::min(remain_pad, 0));
}

int GetIm2colBufferSize(const index_t *out_shape, const index_t *filter_shape) {
  int depth = filter_shape[1] * filter_shape[2] * filter_shape[3];
  int column = out_shape[0] * out_shape[2] * out_shape[3];
  return depth * column;
}

std::vector<index_t> GetIm2colBufferShape(const index_t *out_shape,
                                          const index_t *filter_shape) {
  index_t depth = filter_shape[1] * filter_shape[2] * filter_shape[3];
  index_t column = out_shape[0] * out_shape[2] * out_shape[3];
  return {depth, column};
}

namespace column {

index_t GetIm2colBufferSize(const index_t *out_shape, const int cols_nnz) {
  return out_shape[0] * out_shape[2] * out_shape[3] * cols_nnz;
}

std::vector<index_t> GetIm2colBufferShape(const index_t *out_shape,
                                          const int cols_nnz) {
  return {cols_nnz, out_shape[0] * out_shape[2] * out_shape[3]};
}

} // namespace column

} // namespace im2col
} // namespace deepvan