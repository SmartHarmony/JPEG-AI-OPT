#ifndef DEEPVAN_BACKEND_ARM_FP32_H_
#define DEEPVAN_BACKEND_ARM_FP32_H_

#include "deepvan/backend/arm/fp32/conv_2d.h"
#include "deepvan/backend/arm/fp32/gemm.h"
#include "deepvan/backend/common/im2col.h"
#include "deepvan/core/op_context.h"
#include "deepvan/core/tensor.h"
#include <memory>
#include <vector>

namespace deepvan {
namespace arm {
namespace fp32 {

class Conv2dGEMM : public Conv2dBase {
public:
  Conv2dGEMM(const std::vector<int> strides, const std::vector<int> dilations,
             const std::vector<int> paddings, const Padding padding_type)
      : Conv2dBase(strides, dilations, paddings, padding_type) {}

  virtual ~Conv2dGEMM() {}

  VanState Compute(const OpContext *context, const Tensor *input,
                   const Tensor *filter, Tensor *output) override;

private:
  Gemm gemm_;
};

} // namespace fp32
} // namespace arm
} // namespace deepvan
#endif