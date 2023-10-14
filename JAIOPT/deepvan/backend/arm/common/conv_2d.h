#ifndef DEEPVAN_BACKEND_ARM_COMMON_CONV_2D_H_
#define DEEPVAN_BACKEND_ARM_COMMON_CONV_2D_H_

#include <arm_neon.h>
#include <functional>
#include <memory>
#include <vector>

#include "deepvan/backend/common/conv_pool_2d_util.h"
#include "deepvan/core/op_context.h"
#include "deepvan/core/tensor.h"
#include "deepvan/export/deepvan.h"
#include "deepvan/utils/math.h"

namespace deepvan {
namespace arm {

class Conv2dCommon {
public:
  Conv2dCommon(const std::vector<int> kernels,
               const std::vector<int> strides,
               const std::vector<int> dilations,
               const std::vector<int> paddings,
               const Padding padding_type,
               const std::string name = "")
      : kernels_(kernels),
        strides_(strides),
        dilations_(dilations),
        paddings_(paddings),
        padding_type_(padding_type),
        name_(name) {}

  virtual ~Conv2dCommon() = default;

  virtual VanState Compute(const OpContext *context,
                           const Tensor *input,
                           const Tensor *filter,
                           Tensor *output) = 0;

protected:
  std::vector<int> strides_;
  std::vector<int> dilations_;
  std::vector<int> paddings_;
  Padding padding_type_;
  std::string name_;
  std::vector<int> kernels_;
};

} // namespace arm
} // namespace deepvan

#endif // DEEPVAN_BACKEND_ARM_COMMON_CONV_2D_H_
