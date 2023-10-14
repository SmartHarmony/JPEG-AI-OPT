#ifndef DEEPVAN_BACKEND_OPENCL_DECONV_2D_H_
#define DEEPVAN_BACKEND_OPENCL_DECONV_2D_H_

#include <vector>

#include "deepvan/backend/activation_op.h"

namespace deepvan {
class OpContext;
class Tensor;


class OpenCLDeconv2dKernel {
 public:
  virtual VanState Compute(
      OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      const Tensor *bias,
      const int *strides,
      const int *padding_data,
      const ActivationType activation,
      const float relux_max_limit,
      const float leakyrelu_coefficient,
      const std::vector<index_t> &output_shape,
      Tensor *output) = 0;
  DEEPVAN_EMPTY_VIRTUAL_DESTRUCTOR(OpenCLDeconv2dKernel);
};
}  // namespace deepvan

#endif  // DEEPVAN_BACKEND_OPENCL_DECONV_2D_H_
