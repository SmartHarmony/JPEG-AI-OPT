#ifndef DEEPVAN_BACKEND_OPENCL_CONV_2D_H_
#define DEEPVAN_BACKEND_OPENCL_CONV_2D_H_

#include <vector>

#include "deepvan/backend/common/activation_type.h"
#include "deepvan/backend/common/conv_pool_2d_util.h"

namespace deepvan {
class OpContext;

class OpenCLConv2dKernel {
public:
  virtual bool CheckUseWinograd(OpenCLRuntime *runtime,
                                const std::vector<index_t> &filter_shape,
                                const std::vector<index_t> &output_shape,
                                const int *strides,
                                const int *dilations,
                                int *wino_block_size) = 0;

  virtual VanState Compute(OpContext *context,
                           const Tensor *input,
                           const Tensor *filter,
                           const Tensor *bias,
                           const int *strides,
                           const Padding &padding_type,
                           const std::vector<int> &padding_data,
                           const int *dilations,
                           const ActivationType activation,
                           const float relux_max_limit,
                           const float leakyrelu_coefficient,
                           const int winograd_blk_size,
                           Tensor *output,
                           const Tensor *elt=nullptr) = 0;
  DEEPVAN_EMPTY_VIRTUAL_DESTRUCTOR(OpenCLConv2dKernel);
};

} // namespace deepvan

#endif // DEEPVAN_BACKEND_OPENCL_CONV_2D_H_
