#ifndef DEEPVAN_BACKEND_ARM_FP32_CONV_2D_1X1_H_
#define DEEPVAN_BACKEND_ARM_FP32_CONV_2D_1X1_H_

#include "deepvan/backend/arm/fp32/conv_2d.h"
#include "deepvan/backend/arm/fp32/gemm.h"
#include "deepvan/core/op_context.h"
#include "deepvan/core/tensor.h"
#include "deepvan/export/deepvan.h"
#include <vector>

namespace deepvan {
namespace arm {
namespace fp32 {

class Conv2dK1x1 : public Conv2dBase {
public:
  Conv2dK1x1(const std::vector<int> paddings,
             const Padding padding_type,
             const std::vector<int> strides)
      : Conv2dBase(strides, {1, 1}, paddings, padding_type), gemm_(true) {}
  virtual ~Conv2dK1x1() {}

  VanState Compute(const OpContext *context,
                   const Tensor *input,
                   const Tensor *filter,
                   Tensor *output);

private:
  VanState ComputeK1x1S2(const OpContext *context,
                         const Tensor *input,
                         const Tensor *filter,
                         Tensor *output);
  VanState ComputeK1x1S1(const OpContext *context,
                         const Tensor *input,
                         const Tensor *filter,
                         Tensor *output);

  Gemm gemm_;
};

} // namespace fp32
} // namespace arm
} // namespace deepvan

#endif // DEEPVAN_BACKEND_ARM_FP32_CONV_2D_1X1_H_
