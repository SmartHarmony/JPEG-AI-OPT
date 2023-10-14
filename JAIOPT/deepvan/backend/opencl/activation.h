#ifndef DEEPVAN_BACKEND_OPENCL_ACTIVATION_H_
#define DEEPVAN_BACKEND_OPENCL_ACTIVATION_H_

#include "deepvan/export/deepvan.h"
#include "deepvan/utils/math.h"

namespace deepvan {
class OpContext;
class Tensor;


class OpenCLActivationKernel {
 public:
  virtual VanState Compute(
      OpContext *context,
      const Tensor *input,
      const Tensor *alpha,
      Tensor *output) = 0;
  DEEPVAN_EMPTY_VIRTUAL_DESTRUCTOR(OpenCLActivationKernel);
};

}  // namespace deepvan

#endif  // DEEPVAN_BACKEND_OPENCL_ACTIVATION_H_
