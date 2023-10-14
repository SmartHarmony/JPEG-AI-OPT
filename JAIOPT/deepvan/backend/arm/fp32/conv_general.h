#ifndef DEEPVAN_BACKEND_ARM_FP32_CONV_GENERAL_H_
#define DEEPVAN_BACKEND_ARM_FP32_CONV_GENERAL_H_

#include <vector>
#include "deepvan/export/deepvan.h"
#include "deepvan/core/tensor.h"
#include "deepvan/core/op_context.h"
#include "deepvan/backend/arm/fp32/conv_2d.h"

namespace deepvan {
namespace arm {
namespace fp32 {

class Conv2dGeneral : public Conv2dBase {
 public:
  Conv2dGeneral(const std::vector<int> strides,
                const std::vector<int> dilations,
                const std::vector<int> paddings,
                const Padding padding_type)
      : Conv2dBase(strides, dilations, paddings, padding_type) {}
  virtual ~Conv2dGeneral() {}

  VanState Compute(
      const OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      Tensor *output);
};

}  // namespace fp32
}  // namespace arm
}  // namespace deepvan

#endif  // DEEPVAN_BACKEND_ARM_FP32_CONV_GENERAL_H_
