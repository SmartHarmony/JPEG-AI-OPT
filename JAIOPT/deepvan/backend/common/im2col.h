#ifndef DEEPVAN_BACKEND_COMMON_IM2COL_H_
#define DEEPVAN_BACKEND_COMMON_IM2COL_H_

#include <vector>

#include "deepvan/core/tensor.h"
#include "deepvan/utils/logging.h"

namespace deepvan {
namespace im2col {

bool is_a_ge_zero_and_a_lt_b(int a, int b);

int GetIm2colBufferSize(const index_t *out_shape, const index_t *filter_shape);

std::vector<index_t> GetIm2colBufferShape(const index_t *out_shape,
                                          const index_t *filter_shape);

namespace dense {

// Dense Im2col with strides 1 and dilations 1
template <typename T>
void DenseIm2colNCHWWithS1D1(const T *data_im,
                             const std::vector<index_t> &in_shape,
                             const index_t kernel_h,
                             const index_t kernel_w,
                             const int pad_t,
                             const int pad_l,
                             const std::vector<index_t> &out_shape,
                             T *data_col) {
  const int channels = in_shape[1];
  const int input_h = in_shape[2];
  const int input_w = in_shape[3];
  const int output_h = out_shape[2];
  const int output_w = out_shape[3];
  const int in_channel_size = input_h * input_w;
  const int out_channel_size = output_h * output_w;
  const int kernel_size = kernel_h * kernel_w;
#pragma omp parallel for collapse(2) schedule(runtime)
  for (int channel = 0; channel < channels; channel++) {
    for (int output_row = 0; output_row < output_h; output_row++) {
      auto data_im_ptr = data_im + channel * in_channel_size +
                         std::max(output_row - pad_t, 0) * input_w;
      auto data_col_ptr = data_col +
                          (channel * kernel_size) * out_channel_size +
                          output_row * output_w;
      bool increase_input_ptr;
      int left_zero, right_zero, content;
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        int reduce = 0;
        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          if (kernel_row + output_row < pad_t ||
              kernel_row + output_row >= input_h + pad_t) {
            memset(data_col_ptr, 0, sizeof(T) * output_w);
            increase_input_ptr = false;
          } else {
            left_zero = std::max(pad_l - kernel_col, 0);
            right_zero = std::max(kernel_col - pad_l, 0);
            content = output_w - left_zero - right_zero;
            memset(data_col_ptr, 0, sizeof(T) * left_zero);
            memcpy(data_col_ptr + left_zero, data_im_ptr, sizeof(T) * content);
            memset(
                data_col_ptr + left_zero + content, 0, sizeof(T) * right_zero);
            increase_input_ptr = true;
            if (left_zero == 0) {
              data_im_ptr += 1;
              reduce++;
            }
          }
          data_col_ptr += out_channel_size;
        } // kernel_col
        if (increase_input_ptr) {
          data_im_ptr += input_w - reduce;
        }
      } // kernel_row
    }   // output_row
  }     // channel
}

template <typename T>
void DenseIm2colNCHWCommon(const T *data_im,
                           const std::vector<index_t> &in_shape,
                           const index_t kernel_h,
                           const index_t kernel_w,
                           const index_t stride_h,
                           const index_t stride_w,
                           const int pad_t,
                           const int pad_l,
                           const int dilation_h,
                           const int dilation_w,
                           const std::vector<index_t> &out_shape,
                           T *data_col) {
  const int channels = in_shape[1];
  const int input_h = in_shape[2];
  const int input_w = in_shape[3];
  const int output_h = out_shape[2];
  const int output_w = out_shape[3];
  const int in_channel_size = input_h * input_w;
  const int out_channel_size = output_h * output_w;
#pragma omp parallel for collapse(3) schedule(runtime)
  for (int channel = 0; channel < channels; channel++) {
    for (int kh = 0; kh < kernel_h; kh++) {
      for (int kw = 0; kw < kernel_w; kw++) {
        auto data_col_ptr =
            data_col + (channel * kernel_h * kernel_w + kh * kernel_h + kw) *
                           out_channel_size;
        auto data_im_ptr = data_im + in_channel_size * channel;
        int input_row = -pad_t + kh * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, input_h)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col_ptr++) = 0;
            }
          } else {
            int input_col = -pad_l + kw * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, input_w)) {
                *(data_col_ptr++) =
                    data_im_ptr[input_row * input_w + input_col];
              } else {
                *(data_col_ptr++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        } // output_rows
      }   // kw
    }     // kh
  }       // channel
}

// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
template <typename T>
void DenseIm2colNCHW(const T *data_im,
                     const std::vector<index_t> &in_shape,
                     const index_t kernel_h,
                     const index_t kernel_w,
                     const index_t stride_h,
                     const index_t stride_w,
                     const int pad_t,
                     const int pad_l,
                     const int dilation_h,
                     const int dilation_w,
                     const std::vector<index_t> &out_shape,
                     T *data_col) {
  DenseIm2colNCHWCommon<T>(data_im,
                           in_shape,
                           kernel_h,
                           kernel_w,
                           stride_h,
                           stride_w,
                           pad_t,
                           pad_l,
                           dilation_h,
                           dilation_w,
                           out_shape,
                           data_col);
}

} // namespace dense

namespace column {

index_t GetIm2colBufferSize(const index_t *out_shape, const int cols_nnz);

std::vector<index_t> GetIm2colBufferShape(const index_t *out_shape,
                                          const int cols_nnz);

template <typename T>
void ColumnIm2colNCHWCommon(const T *data_im,
                            const int *cols_mask,
                            const std::vector<index_t> &in_shape,
                            const index_t kernel_h,
                            const index_t kernel_w,
                            const index_t stride_h,
                            const index_t stride_w,
                            const int pad_t,
                            const int pad_l,
                            const int dilation_h,
                            const int dilation_w,
                            const std::vector<index_t> &out_shape,
                            T *data_col) {
  const int channels = in_shape[1];
  const int input_h = in_shape[2];
  const int input_w = in_shape[3];
  const int output_h = out_shape[2];
  const int output_w = out_shape[3];
  const int in_channel_size = input_h * input_w;
  const int out_channel_size = output_h * output_w;
  const int kernel_size = kernel_h * kernel_w;
#pragma omp parallel for collapse(3) schedule(runtime)
  for (int channel = 0; channel < channels; channel++) {
    for (int kh = 0; kh < kernel_h; kh++) {
      for (int kw = 0; kw < kernel_w; kw++) {
        const int kernel_col_idx = channel * kernel_size + kh * kernel_h + kw;
        const bool is_empty_col =
            cols_mask[kernel_col_idx + 1] == cols_mask[kernel_col_idx];
        if (is_empty_col) {
          continue;
        } else {
          auto data_col_ptr =
              data_col + cols_mask[kernel_col_idx] * out_channel_size;
          const auto data_im_ptr = data_im + channel * in_channel_size;
          int input_row = -pad_t + kh * dilation_h;
          for (int output_rows = output_h; output_rows; output_rows--) {
            if (!is_a_ge_zero_and_a_lt_b(input_row, input_h)) {
              for (int output_cols = output_w; output_cols; output_cols--) {
                *(data_col_ptr++) = 0;
              }
            } else {
              int input_col = -pad_l + kw * dilation_w;
              for (int output_col = output_w; output_col; output_col--) {
                if (is_a_ge_zero_and_a_lt_b(input_col, input_w)) {
                  *(data_col_ptr++) =
                      data_im_ptr[input_row * input_w + input_col];
                } else {
                  *(data_col_ptr++) = 0;
                }
                input_col += stride_w;
              }
            }
            input_row += stride_h;
          } // output_rows
        }   // else
      }     // kw
    }       // kh
  }         // channel
}

template <typename T>
void ColumnIm2colNCHW(const T *data_im,
                      const int *cols_mask,
                      const std::vector<index_t> &in_shape,
                      const index_t kernel_h,
                      const index_t kernel_w,
                      const index_t stride_h,
                      const index_t stride_w,
                      const int pad_t,
                      const int pad_l,
                      const int dilation_h,
                      const int dilation_w,
                      const std::vector<index_t> &out_shape,
                      T *data_col) {

  ColumnIm2colNCHWCommon<T>(data_im,
                            cols_mask,
                            in_shape,
                            kernel_h,
                            kernel_w,
                            stride_h,
                            stride_w,
                            pad_t,
                            pad_l,
                            dilation_h,
                            dilation_w,
                            out_shape,
                            data_col);
}

template <typename T>
void PadInputAndIm2colS1(const T *data_im,
                         const int *cols_mask,
                         const int cols_nnz,
                         const std::vector<index_t> &in_shape,
                         const index_t kernel_h,
                         const index_t kernel_w,
                         const int pad_t,
                         const int pad_l,
                         const std::vector<index_t> &out_shape,
                         T *data_col) {
  const int input_h = in_shape[2];
  const int input_w = in_shape[3];
  const int output_h = out_shape[2];
  const int output_w = out_shape[3];
  const int in_channel_size = input_h * input_w;
  const int out_channel_size = output_h * output_w;
  const int kernel_size = kernel_h * kernel_w;

#pragma omp parallel for collapse(2) schedule(runtime)
  for (int col_idx = 0; col_idx < cols_nnz; col_idx++) {
    for (int out_h_idx = 0; out_h_idx < output_h; out_h_idx++) {
      const int kernel_col_idx = cols_mask[col_idx];
      const int ic = kernel_col_idx / kernel_size;
      const int kernel_inner_idx = kernel_col_idx % kernel_size;
      const int kh = kernel_inner_idx / kernel_h;
      const int kw = kernel_inner_idx % kernel_h;

      int input_row = -pad_t + kh + out_h_idx;
      int input_col = -pad_l + kw;
      const int input_offset = ic * in_channel_size +
                               std::max(0, input_row) * input_w +
                               std::max(0, input_col);
      const int output_offset =
          col_idx * out_channel_size + out_h_idx * output_w;

      if (input_row < 0 || input_row >= input_h) {
        memset(data_col, 0, output_w * sizeof(T));
      } else {
        int left, content, right;
        if (input_col < 0) {
          left = -input_col;
          content = std::min(input_w - left, output_w - left);
          right = std::max(0, output_w - content - left);
        } else {
          left = 0;
          content = std::min(input_w - input_col, output_w);
          right = std::max(0, output_w - content);
        }
        int offset = 0;
        if (left > 0) {
          memset(data_col + output_offset, 0, left * sizeof(T));
        }
        offset += left;
        memcpy(data_col + output_offset + offset,
               data_im + input_offset,
               content * sizeof(T));
        offset += content;
        if (right > 0) {
          memset(data_col + output_offset + offset, 0, right * sizeof(T));
        }
      }
    } // h
  }   // col_idx
}

template <typename T>
void PadInputAndIm2colS2(const T *data_im,
                         const int *cols_mask,
                         const int cols_nnz,
                         const std::vector<index_t> &in_shape,
                         const index_t kernel_h,
                         const index_t kernel_w,
                         const int pad_t,
                         const int pad_l,
                         const std::vector<index_t> &out_shape,
                         T *data_col) {
  const int input_h = in_shape[2];
  const int input_w = in_shape[3];
  const int output_h = out_shape[2];
  const int output_w = out_shape[3];
  const int in_channel_size = input_h * input_w;
  const int out_channel_size = output_h * output_w;
  const int kernel_size = kernel_h * kernel_w;

#pragma omp parallel for collapse(1) schedule(runtime)
  for (int col_idx = 0; col_idx < cols_nnz; col_idx++) {
    const int kernel_col_idx = cols_mask[col_idx];
    const int ic = kernel_col_idx / kernel_size;
    const int kernel_inner_idx = kernel_col_idx % kernel_size;
    const int kh = kernel_inner_idx / kernel_h;
    const int kw = kernel_inner_idx % kernel_h;

    auto data_col_ptr = data_col + col_idx * out_channel_size;
    const auto data_im_ptr = data_im + ic * in_channel_size;
    int input_row = -pad_t + kh;
    for (int output_rows = output_h; output_rows; output_rows--) {
      if (!is_a_ge_zero_and_a_lt_b(input_row, input_h)) {
        for (int output_cols = output_w; output_cols; output_cols--) {
          *(data_col_ptr++) = 0;
        }
      } else {
        int input_col = -pad_l + kw;
        for (int output_col = output_w; output_col; output_col--) {
          if (is_a_ge_zero_and_a_lt_b(input_col, input_w)) {
            *(data_col_ptr++) = data_im_ptr[input_row * input_w + input_col];
          } else {
            *(data_col_ptr++) = 0;
          }
          input_col += 2;
        }
      }
      input_row += 2;
    } // output_rows
  }   // col_idx
}

} // namespace column
} // namespace im2col
} // namespace deepvan

#endif // DEEPVAN_BACKEND_COMMON_IM2COL_H_