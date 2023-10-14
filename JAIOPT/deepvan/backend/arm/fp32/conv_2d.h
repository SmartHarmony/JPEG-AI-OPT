#ifndef DEEPVAN_BACKEND_ARM_FP32_CONV_2D_H_
#define DEEPVAN_BACKEND_ARM_FP32_CONV_2D_H_

#include <arm_neon.h>
#include <memory>
#include <vector>

#include "deepvan/backend/arm/common/conv_2d.h"
#include "deepvan/backend/arm/fp32/gemm.h"
#include "deepvan/backend/common/conv_pool_2d_util.h"
#include "deepvan/core/op_context.h"
#include "deepvan/core/tensor.h"
#include "deepvan/export/deepvan.h"

namespace deepvan {
namespace arm {
namespace fp32 {

class Conv2dBase : public Conv2dCommon{
public:
  Conv2dBase(const std::vector<int> strides,
             const std::vector<int> dilations,
             const std::vector<int> paddings,
             const Padding padding_type,
             const std::string name = "")
      : Conv2dCommon({0, 0}, strides, dilations, paddings, padding_type, name) {
  }

  virtual ~Conv2dBase() = default;

  virtual VanState Compute(const OpContext *context,
                           const Tensor *input,
                           const Tensor *filter,
                           Tensor *output) = 0;

protected:
  void CalOutputShapeAndInputPadSize(const std::vector<index_t> &input_shape,
                                     const std::vector<index_t> &filter_shape,
                                     std::vector<index_t> *output_shape,
                                     std::vector<int> *in_pad_size);

  void CalOutputBoundaryWithoutUsingInputPad(
      const std::vector<index_t> &output_shape,
      const std::vector<int> in_pad_size,
      std::vector<index_t> *out_bound);

  void CalOutputShapeAndPadSize(const Tensor *input,
                                const Tensor *filter,
                                const int out_tile_height,
                                const int out_tile_width,
                                std::vector<index_t> *output_shape,
                                std::vector<int> *in_pad_size,
                                std::vector<int> *out_pad_size);

  VanState ResizeOutAndPadInOut(const OpContext *context,
                                const Tensor *input,
                                const Tensor *filter,
                                Tensor *output,
                                const int out_tile_height,
                                const int out_tile_width,
                                std::unique_ptr<const Tensor> *padded_input,
                                std::unique_ptr<Tensor> *padded_output);

  void PadInput(const Tensor &src,
                const int pad_top,
                const int pad_left,
                Tensor *dst);
  void UnPadOutput(const Tensor &src, Tensor *dst);

  void StrideInput(const Tensor &src, Tensor *dst);
};

} // namespace fp32
} // namespace arm
} // namespace deepvan

#endif // DEEPVAN_BACKEND_ARM_FP32_CONV_2D_H_
