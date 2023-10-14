#ifndef DEEPVAN_BACKEND_ARM_FP32_CONV_2D_KERNEL_H_
#define DEEPVAN_BACKEND_ARM_FP32_CONV_2D_KERNEL_H_

#include "deepvan/export/deepvan.h"
#include "deepvan/core/tensor.h"
#include "deepvan/core/op_context.h"
#include "deepvan/backend/arm/fp32/gemm.h"
#ifdef NEON_SUPPORT
#include <arm_neon.h>
#endif

namespace deepvan {
namespace arm {
namespace fp32 {

namespace {
  typedef struct conv2d_sparse_point {
    int col;
    int row;
    index_t offset;
    float value;
    const float *input_ptr;
    const float *input_ptr_next;
    conv2d_sparse_point(int idx, 
                        float v, 
                        index_t w,
                        const float *ptr):col(idx % 3), row(idx / 3), offset(w * row + col), value(v) {
      input_ptr = ptr + offset;
      input_ptr_next = input_ptr + w;
    }
  } Conv2dSparsePoint;

  //sparse点的封装，只关注点的位置
  typedef struct conv2d_point {
    int col, row;
    conv2d_point(int col_index) {
      row = col_index / 3;
      col = col_index % 3;
    }
  } Conv2dPoint;
  
  
}//end namespace

class Conv2dKernelBase {
 public:
  Conv2dKernelBase() = delete;
  explicit Conv2dKernelBase(const Tensor *filter, 
                            const int *stride, 
                            const int *dilation) {
    filter_ = filter;
    strides_ = stride;
    dilations_ = dilation;
  }

  virtual ~Conv2dKernelBase() = default;
  
  virtual VanState Compute(
      const OpContext *context,
      const float *input,
      const index_t *extra_in_shape,
      const index_t *extra_out_shape,
      float *output) = 0;
 
 protected:
  virtual VanState ComputeDense(const OpContext *context,
                                  const float *input,
                                  const index_t *extra_in_shape,
                                  const index_t *extra_out_shape,
                                  float *output) = 0;
  
  virtual VanState ComputeSparse(const OpContext *context,
                                   const float *input,
                                   const index_t *extra_in_shape,
                                   const index_t *extra_out_shape,
                                   float *output) = 0;
  const Tensor *filter_;
  const int *strides_;
  const int *dilations_;
};

// calculate one output channel and one input channel
inline void Conv2dCPUKHxKWCalc(const float *in_ptr,
                               const float *filter_ptr,
                               const index_t in_width,
                               const index_t filter_height,
                               const index_t filter_width,
                               const index_t out_height,
                               const index_t out_width,
                               float *out_ptr,
                               const int stride) {
  for (index_t h = 0; h < out_height; ++h) {
    for (index_t w = 0; w < out_width; ++w) {
      for (int i = 0; i < filter_height; ++i) {
        for (int j = 0; j < filter_width; ++j) {
          out_ptr[h * out_width + w] +=
              in_ptr[(h * stride + i) * in_width + (w * stride + j)] *
              filter_ptr[i * filter_width + j];
        }
      }
    }
  }
}

}  // namespace fp32
}  // namespace arm
}  // namespace deepvan

#endif  // DEEPVAN_BACKEND_ARM_FP32_CONV_2D_KERNEL_H_
