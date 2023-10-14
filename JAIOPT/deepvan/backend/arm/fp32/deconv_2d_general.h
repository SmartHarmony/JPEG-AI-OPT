#ifndef DEEPVAN_BACKEND_ARM_FP32_DECONV_2D_GENERAL_H_
#define DEEPVAN_BACKEND_ARM_FP32_DECONV_2D_GENERAL_H_

#include <memory>
#include <vector>

#include "deepvan/backend/arm/fp32/deconv_2d.h"
#include "deepvan/backend/arm/fp32/gemm.h"
#include "deepvan/backend/common/conv_pool_2d_util.h"
#include "deepvan/core/op_context.h"
#include "deepvan/core/tensor.h"
#include "deepvan/core/types.h"
#include "deepvan/export/deepvan.h"
#include "deepvan/utils/memory.h"

namespace deepvan {
namespace arm {
namespace fp32 {

class Deconv2dGeneral : public Deconv2dBase {
public:
  Deconv2dGeneral(const std::vector<int> &strides,
                  const std::vector<int> &dilations,
                  const std::vector<int> &paddings,
                  const Padding padding_type,
                  const FrameworkType framework_type)
      : Deconv2dBase(strides,
                     dilations,
                     paddings,
                     padding_type,
                     framework_type) {}
  virtual ~Deconv2dGeneral() {}

  VanState Compute(const OpContext *context,
                   const Tensor *input,
                   const Tensor *filter,
                   const Tensor *output_shape,
                   Tensor *output) override;
};

} // namespace fp32
} // namespace arm
} // namespace deepvan

#endif // DEEPVAN_BACKEND_ARM_FP32_DECONV_2D_GENERAL_H_
