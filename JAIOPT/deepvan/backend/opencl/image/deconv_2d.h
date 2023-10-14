#ifndef DEEPVAN_BACKEND_OPENCL_IMAGE_DECONV_2D_H_
#define DEEPVAN_BACKEND_OPENCL_IMAGE_DECONV_2D_H_

#include "deepvan/backend/opencl/deconv_2d.h"
#include "deepvan/backend/opencl/image/image_tensor_debug.h"

#include <memory>
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
class Deconv2dKernel : public OpenCLDeconv2dKernel {
public:
  Deconv2dKernel(std::string name = "") : op_name_(name) {}

  VanState Compute(OpContext *context,
                   const Tensor *input,
                   const Tensor *filter,
                   const Tensor *bias,
                   const int *strides,
                   const int *padding_data,
                   const ActivationType activation,
                   const float relux_max_limit,
                   const float leakyrelu_coefficient,
                   const std::vector<index_t> &output_shape,
                   Tensor *output) override;

private:
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
  std::string op_name_;
};

template <typename T>
VanState Deconv2dKernel<T>::Compute(OpContext *context,
                                    const Tensor *input,
                                    const Tensor *filter,
                                    const Tensor *bias,
                                    const int *strides,
                                    const int *padding_data,
                                    const ActivationType activation,
                                    const float relux_max_limit,
                                    const float leakyrelu_coefficient,
                                    const std::vector<index_t> &output_shape,
                                    Tensor *output) {
  std::vector<size_t> output_image_shape;
  OpenCLUtil::CalImage2DShape(
      output_shape, OpenCLBufferType::IN_OUT_CHANNEL, &output_image_shape, 0, context->model_type());
  RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));
  const DataType dt = DataTypeToEnum<T>::value;
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);
  const index_t input_channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t input_channel_blocks = RoundUpDiv4(input_channels);
  const int stride_h = strides[0];
  const int stride_w = strides[1];
  CONDITIONS(stride_w > 0 && stride_h > 0, "strides should be > 0.");
  const int width_tile = 5;
  const index_t n_strides = (width + stride_w - 1) / stride_w;
  const index_t width_blocks =
      ((n_strides + width_tile - 1) / width_tile) * stride_w;
  const float stride_h_r = 1.f / static_cast<float>(stride_h);
  const float stride_w_r = 1.f / static_cast<float>(stride_w);
  const int padding_h = (padding_data[0] + 1) >> 1;
  const int padding_w = (padding_data[1] + 1) >> 1;

  const int align_h = stride_h - 1 - padding_h;
  const int align_w = stride_w - 1 - padding_w;
  const int kernel_size = filter->dim(2) * filter->dim(3);

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG;
    NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = DEEPVAN_OBFUSCATE_SYMBOL("deconv_2d");
    built_options.emplace("-Ddeconv_2d=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
    built_options.emplace(bias != nullptr ? "-DBIAS" : "");
    switch (activation) {
    case NOOP: break;
    case RELU: built_options.emplace("-DUSE_RELU"); break;
    case RELUX: built_options.emplace("-DUSE_RELUX"); break;
    case TANH: built_options.emplace("-DUSE_TANH"); break;
    case SIGMOID: built_options.emplace("-DUSE_SIGMOID"); break;
    case LEAKYRELU: built_options.emplace("-DUSE_LEAKYRELU"); break;
    case ROUND:  built_options.emplace("-DUSE_ROUND"); break;
    default: LOG(FATAL) << "Unknown activation type: " << activation;
    }

    RETURN_IF_ERROR(runtime->BuildKernel(
        "deconv_2d", kernel_name, built_options, &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width_blocks),
                           static_cast<uint32_t>(height * batch)};

  OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARGS(kernel_);
    SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, *(filter->opencl_image()));
    if (bias != nullptr) {
      kernel_.setArg(idx++, *(bias->opencl_image()));
    }
    kernel_.setArg(idx++, *(output->opencl_image()));
    kernel_.setArg(idx++, relux_max_limit);
    kernel_.setArg(idx++, leakyrelu_coefficient);
    kernel_.setArg(idx++, static_cast<int32_t>(input->dim(1)));
    kernel_.setArg(idx++, static_cast<int32_t>(input->dim(2)));
    kernel_.setArg(idx++, static_cast<int32_t>(input->dim(3)));
    kernel_.setArg(idx++, static_cast<int32_t>(height));
    kernel_.setArg(idx++, static_cast<int32_t>(width));
    kernel_.setArg(idx++, static_cast<int32_t>(channels));
    kernel_.setArg(idx++, static_cast<int32_t>(stride_h));
    kernel_.setArg(idx++, static_cast<int32_t>(stride_w));
    kernel_.setArg(idx++, stride_h_r);
    kernel_.setArg(idx++, stride_w_r);
    kernel_.setArg(idx++, static_cast<int32_t>(align_h));
    kernel_.setArg(idx++, static_cast<int32_t>(align_w));
    kernel_.setArg(idx++, static_cast<int32_t>(padding_h));
    kernel_.setArg(idx++, static_cast<int32_t>(padding_w));
    kernel_.setArg(idx++, static_cast<int32_t>(filter->dim(2)));
    kernel_.setArg(idx++, static_cast<int32_t>(filter->dim(3)));
    kernel_.setArg(idx++, static_cast<int32_t>(kernel_size));
    kernel_.setArg(idx++, static_cast<int32_t>(input_channel_blocks));
    kernel_.setArg(idx++, static_cast<int32_t>(channel_blocks));

    input_shape_ = input->shape();
  }

  // const std::vector<uint32_t> lws = {2, 2, 2, 0};
  std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_);
  std::string tuning_key = Concat("deconv2d_opencl_kernel_",
                                  activation,
                                  output->dim(0),
                                  output->dim(1),
                                  output->dim(2),
                                  output->dim(3));
  RETURN_IF_ERROR(Run3DKernel(runtime, kernel_, gws, lws, context->future()));
      // runtime, kernel_, tuning_key, gws, lws, context->future()));

  // std::vector<size_t> output_pitch = {0, 0};
  // image::WriteNHWC4DResult<float>(output, output_pitch, op_name_, true);
  OUT_OF_RANGE_VALIDATION;
  return VanState::SUCCEED;
}

} // namespace image
} // namespace opencl
} // namespace deepvan

#endif // DEEPVAN_BACKEND_OPENCL_IMAGE_DECONV_2D_H_
