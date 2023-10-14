#ifndef DEEPVAN_BACKEND_CONV_POOL_2D_OP_BASE_H_
#define DEEPVAN_BACKEND_CONV_POOL_2D_OP_BASE_H_

#include <vector>

#include "deepvan/core/operator.h"
#include "deepvan/backend/common/conv_pool_2d_util.h"

namespace deepvan {

class ConvPool2dOpBase : public Operation {
 public:
  explicit ConvPool2dOpBase(OpConstructContext *context)
      : Operation(context),
        strides_(Operation::GetRepeatedArgs<int>("strides")),
        padding_type_(static_cast<Padding>(Operation::GetOptionalArg<int>(
            "padding", static_cast<int>(SAME)))),
        paddings_(Operation::GetRepeatedArgs<int>("padding_values")),
        dilations_(Operation::GetRepeatedArgs<int>("dilations", {1, 1})) {}

 protected:
  std::vector<int> strides_;
  Padding padding_type_;
  std::vector<int> paddings_;
  std::vector<int> dilations_;
};

}  // namespace deepvan

#endif  // DEEPVAN_BACKEND_CONV_POOL_2D_OP_BASE_H_
