#ifndef DEEPVAN_BACKEND_OPENCL_BUFFER_TRANSFORM_KERNEL_H_
#define DEEPVAN_BACKEND_OPENCL_BUFFER_TRANSFORM_KERNEL_H_

#include "deepvan/core/runtime/opencl/opencl_shape_util.h"
#include "deepvan/export/deepvan.h"
#include "deepvan/utils/math.h"

namespace deepvan {
class OpContext;
class Tensor;

class OpenCLBufferTransformKernel {
public:
  virtual VanState Compute(OpContext *context,
                           const Tensor *input,
                           const OpenCLBufferType type,
                           const int wino_blk_size,
                           Tensor *output) = 0;
  DEEPVAN_EMPTY_VIRTUAL_DESTRUCTOR(OpenCLBufferTransformKernel)
};
} // namespace deepvan

#endif // DEEPVAN_BACKEND_OPENCL_BUFFER_TRANSFORM_KERNEL_H_
