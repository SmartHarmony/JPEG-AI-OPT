#ifndef DEEPVAN_BACKEND_OPENCL_IMAGE_BUFFER_TO_IMAGE_C3D_H_
#define DEEPVAN_BACKEND_OPENCL_IMAGE_BUFFER_TO_IMAGE_C3D_H_

#include "deepvan/backend/opencl/buffer_transform_kernel.h"

#include <set>
#include <string>
#include <vector>

#include "deepvan/backend/opencl/helper.h"
#include "deepvan/core/op_context.h"
#include "deepvan/core/tensor.h"

namespace deepvan {
namespace opencl {
namespace image {

template <typename T>
class BufferToImageC3D : public OpenCLBufferTransformKernel {
public:
  VanState Compute(OpContext *context,
                   const Tensor *input,
                   const OpenCLBufferType type,
                   const int wino_blk_size,
                   Tensor *output) override;

private:
  cl::Kernel kernel_;
  std::vector<index_t> input_shape_;
};

template <typename T>
VanState BufferToImageC3D<T>::Compute(OpContext *context,
                                      const Tensor *input,
                                      const OpenCLBufferType type,
                                      const int wino_blk_size,
                                      Tensor *output) {
  auto formatted_buffer_shape = input->shape();
  std::vector<size_t> image_shape;
  OpenCLUtil::CalImage2DShape(
      formatted_buffer_shape, type, &image_shape, wino_blk_size, context->model_type());
  RETURN_IF_ERROR(output->ResizeImage(input->shape(), image_shape));

  uint32_t gws[2] = {static_cast<uint32_t>(image_shape[0]),
                     static_cast<uint32_t>(image_shape[1])};
  std::string kernel_name;
  switch (type) {
  case CONV3D_FILTER: kernel_name = "filter_buffer_to_image"; break;
  case IN_OUT_CHANNEL: kernel_name = "in_out_buffer_to_image"; break;
  case ARGUMENT: kernel_name = "arg_buffer_to_image"; break;
  default: STUB; break;
  }

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::string obfuscated_kernel_name = DEEPVAN_OBFUSCATE_SYMBOL(kernel_name);
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG;
    NON_UNIFORM_WG_CONFIG;
    std::stringstream kernel_name_ss;
    kernel_name_ss << "-D" << kernel_name << "=" << obfuscated_kernel_name;
    built_options.emplace(kernel_name_ss.str());
    if (input->dtype() == output->dtype()) {
      built_options.emplace("-DDATA_TYPE=" +
                            DtToCLDt(DataTypeToEnum<T>::value));
      built_options.emplace("-DCMD_DATA_TYPE=" +
                            DtToCLCMDDt(DataTypeToEnum<T>::value));
    } else {
      built_options.emplace("-DDATA_TYPE=" +
                            DtToUpCompatibleCLDt(DataTypeToEnum<T>::value));
      built_options.emplace("-DCMD_DATA_TYPE=" +
                            DtToUpCompatibleCLCMDDt(DataTypeToEnum<T>::value));
    }
    RETURN_IF_ERROR(runtime->BuildKernel("buffer_to_image_c3d",
                                         obfuscated_kernel_name,
                                         built_options,
                                         &kernel_));
  }

  OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARGS(kernel_);
    SET_2D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_buffer()));
    CONDITIONS(input->buffer_offset() % GetEnumTypeSize(input->dtype()) == 0,
               "buffer offset not aligned");
    kernel_.setArg(idx++,
                   static_cast<uint32_t>(input->buffer_offset() /
                                         GetEnumTypeSize(input->dtype())));
    if (type == CONV3D_FILTER) {
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(0)));
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(1)));
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(2)));
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(3)));
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(4)));
    } else if (type == ARGUMENT) {
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(0)));
    } else {
      // IN_OUT_CHANNEL
      kernel_.setArg(idx++, static_cast<uint32_t>(formatted_buffer_shape[3]));
      kernel_.setArg(idx++, static_cast<uint32_t>(formatted_buffer_shape[4]));
      kernel_.setArg(idx++, static_cast<uint32_t>(formatted_buffer_shape[2]));
      kernel_.setArg(idx++, static_cast<uint32_t>(formatted_buffer_shape[1]));
    }
    kernel_.setArg(idx++, *(output->opencl_image()));
    input_shape_ = input->shape();
  }

  const uint32_t kwg_size =
      static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  uint32_t lws0 = 16;
  uint32_t lws1 = kwg_size / 16;
  const std::vector<uint32_t> lws = {lws0, lws1};
  RETURN_IF_ERROR(Run2DKernel(runtime, kernel_, gws, lws, context->future()));
  OUT_OF_RANGE_VALIDATION;
  return VanState::SUCCEED;
}

} // namespace image
} // namespace opencl
} // namespace deepvan

#endif // DEEPVAN_BACKEND_OPENCL_IMAGE_BUFFER_TO_IMAGE_C3D_H_
