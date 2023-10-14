#ifndef DEEPVAN_BACKEND_OPENCL_KERNEL_UTIL_H_
#define DEEPVAN_BACKEND_OPENCL_KERNEL_UTIL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "deepvan/core/op_context.h"
#include "deepvan/core/tensor.h"
#include "deepvan/backend/opencl/helper.h"
#include "deepvan/backend/common/conv_pool_2d_util.h"
#include "deepvan/backend/common/activation_type.h"
#include "deepvan/utils/math.h"

namespace deepvan {
namespace opencl {

VanState ChannelReorder(OpContext *context,
                          cl::Kernel &kernel_,
                          bool &init_pad_kernel,
                          const Tensor *src,
                          const Tensor *filter,
                          std::vector<int> &paddings,
                          Tensor *dst,
                          StatsFuture *future);

} //opencl

} //deepvan


#endif //DEEPVAN_BACKEND_OPENCL_KERNEL_UTIL_H_