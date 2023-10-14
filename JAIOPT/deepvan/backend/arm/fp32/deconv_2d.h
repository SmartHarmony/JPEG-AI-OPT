#ifndef DEEPVAN_BACKEND_ARM_FP32_DCONV_2D_H_
#define DEEPVAN_BACKEND_ARM_FP32_DCONV_2D_H_

#include <vector>
#include <utility>
#include <functional>

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

class Deconv2dBase {
public:
  Deconv2dBase(const std::vector<int> &strides,
               const std::vector<int> &dilations,
               const std::vector<int> &paddings,
               const Padding padding_type,
               const index_t group,
               const FrameworkType framework_type)
      : strides_(strides),
        dilations_(dilations),
        paddings_(paddings),
        padding_type_(padding_type),
        group_(group),
        framework_type_(framework_type) {}

  Deconv2dBase(const std::vector<int> &strides,
               const std::vector<int> &dilations,
               const std::vector<int> &paddings,
               const Padding padding_type,
               const FrameworkType framework_type)
      : Deconv2dBase(strides,
                     dilations,
                     paddings,
                     padding_type,
                     1,
                     framework_type) {}

  virtual ~Deconv2dBase() = default;

  virtual VanState Compute(const OpContext *context,
                             const Tensor *input,
                             const Tensor *filter,
                             const Tensor *output_shape,
                             Tensor *output) = 0;

protected:
  VanState ResizeOutAndPadOut(const OpContext *context,
                                const Tensor *input,
                                const Tensor *filter,
                                const Tensor *output_shape,
                                Tensor *output,
                                std::vector<int> *out_pad_size,
                                std::unique_ptr<Tensor> *padded_output);

  void UnPadOutput(const Tensor &src,
                   const std::vector<int> &out_pad_size,
                   Tensor *dst);

  const std::vector<int> strides_;
  const std::vector<int> dilations_;
  const std::vector<int> paddings_;
  const Padding padding_type_;
  index_t group_;
  const FrameworkType framework_type_;
};

} // namespace fp32
} // namespace arm
} // namespace deepvan

#endif // DEEPVAN_BACKEND_ARM_FP32_DCONV_2D_H_
