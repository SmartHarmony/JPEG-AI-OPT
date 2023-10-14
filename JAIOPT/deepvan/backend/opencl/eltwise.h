#ifndef DEEPVAN_BACKEND_OPENCL_ELTWISE_H_
#define DEEPVAN_BACKEND_OPENCL_ELTWISE_H_

#include "deepvan/export/deepvan.h"
#include "deepvan/utils/math.h"

namespace deepvan {
class OpContext;
class Tensor;


class OpenCLEltwiseKernel {
 public:
  virtual VanState Compute(
      OpContext *context,
      const Tensor *input0,
      const Tensor *input1,
      Tensor *output) = 0;
  DEEPVAN_EMPTY_VIRTUAL_DESTRUCTOR(OpenCLEltwiseKernel);
};

}  // namespace deepvan

#endif  // DEEPVAN_BACKEND_OPENCL_ELTWISE_H_
