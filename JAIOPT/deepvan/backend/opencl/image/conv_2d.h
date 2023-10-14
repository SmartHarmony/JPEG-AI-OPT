#ifndef DEEPVAN_BACKEND_OPENCL_IMAGE_CONV_2D_H_
#define DEEPVAN_BACKEND_OPENCL_IMAGE_CONV_2D_H_

#include "deepvan/backend/opencl/conv_2d.h"
#include "deepvan/backend/opencl/image/image_tensor_debug.h"
#include <memory>
#include <vector>

#include "deepvan/backend/opencl/helper.h"
#include "deepvan/core/op_context.h"
#include "deepvan/core/tensor.h"

namespace deepvan {
namespace opencl {
namespace image {

extern VanState Conv2dK1x1(OpContext *context,
                           cl::Kernel *kernel,
                           const Tensor *input,
                           const Tensor *filter,
                           const Tensor *bias,
                           const int stride,
                           const int *padding,
                           const int *dilations,
                           const ActivationType activation,
                           const float relux_max_limit,
                           const float leakyrelu_coefficient,
                           const DataType dt,
                           std::vector<index_t> *prev_input_shape,
                           Tensor *output,
                           uint32_t *kwg_size,
                           const std::string &op_name,
                           const Tensor *elt);

extern VanState Conv2dK3x3(OpContext *context,
                           cl::Kernel *kernel,
                           const Tensor *input,
                           const Tensor *filter,
                           const Tensor *bias,
                           const int stride,
                           const int *padding,
                           const int *dilations,
                           const ActivationType activation,
                           const float relux_max_limit,
                           const float leakyrelu_coefficient,
                           const DataType dt,
                           std::vector<index_t> *prev_input_shape,
                           Tensor *output,
                           uint32_t *kwg_size,
                           const std::string &op_name,
                           const Tensor *elt);

extern VanState Conv2d(OpContext *context,
                       cl::Kernel *kernel,
                       const Tensor *input,
                       const Tensor *filter,
                       const Tensor *bias,
                       const int stride,
                       const int *padding,
                       const int *dilations,
                       const ActivationType activation,
                       const float relux_max_limit,
                       const float leakyrelu_coefficient,
                       const DataType dt,
                       std::vector<index_t> *prev_input_shape,
                       Tensor *output,
                       uint32_t *kwg_size,
                       const std::string &op_name,
                       const Tensor *elt);

extern VanState WinogradConv2dK3x3S1(OpContext *context,
                                     cl::Kernel *kernels[3],
                                     const Tensor *input,
                                     const Tensor *filter,
                                     const Tensor *bias,
                                     const int *padding,
                                     const ActivationType activation,
                                     const float relux_max_limit,
                                     const float leakyrelu_coefficient,
                                     const DataType dt,
                                     const int wino_blk_size,
                                     std::vector<index_t> *prev_input_shape,
                                     Tensor *output,
                                     uint32_t *kwg_size[3],
                                     const std::string &op_name);

template <typename T>
class Conv2dKernel : public OpenCLConv2dKernel {
public:
  Conv2dKernel(std::string name = "") : op_name_(name) {}

  bool CheckUseWinograd(OpenCLRuntime *runtime,
                        const std::vector<index_t> &filter_shape,
                        const std::vector<index_t> &output_shape,
                        const int *strides,
                        const int *dilations,
                        int *wino_block_size) override;

  VanState Compute(OpContext *context,
                   const Tensor *input,
                   const Tensor *filter,
                   const Tensor *bias,
                   const int *strides,
                   const Padding &padding_type,
                   const std::vector<int> &padding_data,
                   const int *dilations,
                   const ActivationType activation,
                   const float relux_max_limit,
                   const float leakyrelu_coefficient,
                   const int wino_blk_size,
                   Tensor *output,
                   const Tensor *elt = nullptr) override;

private:
  cl::Kernel kernels_[3];
  uint32_t kwg_size_[3];
  std::vector<index_t> input_shape_;
  std::string op_name_;
};

template <typename T>
bool Conv2dKernel<T>::CheckUseWinograd(
    OpenCLRuntime *runtime,
    const std::vector<deepvan::index_t> &filter_shape,
    const std::vector<deepvan::index_t> &output_shape,
    const int *strides,
    const int *dilations,
    int *wino_blk_size) {
  if (filter_shape[2] != 3 || filter_shape[3] != 3 || strides[0] > 1 ||
      strides[1] > 1 || dilations[0] > 1 || dilations[1] > 1) {
    return false;
  }
  index_t out_channels = filter_shape[0];
  index_t in_channels = filter_shape[1];
  auto opencl_image_max_size = runtime->GetMaxImage2DSize();
  auto check_opencl_limit = [&](int block_size) -> bool {
    int sqr_block = (block_size + 2) * (block_size + 2);
    uint64_t transformed_width = static_cast<uint64_t>(
        output_shape[0] * ((output_shape[1] + block_size - 1) / block_size) *
        ((output_shape[2] + block_size - 1) / block_size));
    VLOG(INFO) << "opencl_image_max_size: " << MakeString(opencl_image_max_size)
               << ", transformed_width: " << transformed_width
               << ", sqr_block * in_channels: "
               << static_cast<uint64_t>(sqr_block * in_channels)
               << ", sqr_block * out_channels: "
               << static_cast<uint64_t>(sqr_block * out_channels)
               << ", in_channels: " << in_channels
               << ", out_channels: " << out_channels;
    return (transformed_width < opencl_image_max_size[0] &&
            static_cast<uint64_t>(sqr_block * in_channels) <
                opencl_image_max_size[1] &&
            static_cast<uint64_t>(sqr_block * out_channels) <
                opencl_image_max_size[1]);
  };
  // GPU only supports 4x4 and 2x2 gpu winograd convolution
  if (*wino_blk_size == 4) {
    // if block size == 4 exceed OpenCL image size limitation, fallback to 2
    if (!check_opencl_limit(4)) {
      *wino_blk_size = 2;
    } else {
      return true;
    }
  }
  return check_opencl_limit(2);
}

template <typename T>
VanState Conv2dKernel<T>::Compute(OpContext *context,
                                  const Tensor *input,
                                  const Tensor *filter,
                                  const Tensor *bias,
                                  const int *strides,
                                  const Padding &padding_type,
                                  const std::vector<int> &padding_data,
                                  const int *dilations,
                                  const ActivationType activation,
                                  const float relux_max_limit,
                                  const float leakyrelu_coefficient,
                                  const int wino_blk_size,
                                  Tensor *output,
                                  const Tensor *elt) {
  index_t kernel_h = filter->dim(2);
  index_t kernel_w = filter->dim(3);
  if (strides[0] != strides[1] ||
      (dilations[0] > 1 && (strides[0] > 1 || kernel_h == 1))) {
    LOG(WARNING) << "OpenCL conv2d kernel with "
                 << "filter" << kernel_h << "x" << kernel_w << ","
                 << " stride " << strides[0] << "x" << strides[1]
                 << ",dilations " << dilations[0] << "x" << dilations[1]
                 << " is not implemented yet.";
    UNSUPPORTED_OP("DepthWiseConv2D");
  }

  // Reshape output
  std::vector<index_t> output_shape(4);
  std::vector<int> paddings(2);
  if (padding_data.empty()) {
    CalcNHWCPaddingAndOutputSize(input->shape().data(),
                                 filter->shape().data(),
                                 dilations,
                                 strides,
                                 padding_type,
                                 output_shape.data(),
                                 paddings.data());
  } else {
    paddings = padding_data;
    CalcOutputSize(input->shape().data(),
                   filter->shape().data(),
                   padding_data.data(),
                   dilations,
                   strides,
                   RoundType::FLOOR,
                   output_shape.data());
  }

  std::vector<size_t> output_image_shape;
  OpenCLUtil::CalImage2DShape(
      output_shape, OpenCLBufferType::IN_OUT_CHANNEL, &output_image_shape, wino_blk_size);
  RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));

  std::function<VanState()> conv_func;

  if (wino_blk_size != 0) {
    // use winograd covolution
    conv_func = [&]() -> VanState {
      cl::Kernel *kernels[3] = {&kernels_[0], &kernels_[1], &kernels_[2]};
      uint32_t *kwg_size[3] = {&kwg_size_[0], &kwg_size_[1], &kwg_size_[2]};
      return WinogradConv2dK3x3S1(context,
                                  kernels,
                                  input,
                                  filter,
                                  bias,
                                  paddings.data(),
                                  activation,
                                  relux_max_limit,
                                  leakyrelu_coefficient,
                                  DataTypeToEnum<T>::value,
                                  wino_blk_size,
                                  &input_shape_,
                                  output,
                                  kwg_size,
                                  op_name_);
    };
  } else if (kernel_h == 1 && kernel_w == 1) {
    conv_func = [&]() -> VanState {
      return Conv2dK1x1(context,
                        &kernels_[0],
                        input,
                        filter,
                        bias,
                        strides[0],
                        paddings.data(),
                        dilations,
                        activation,
                        relux_max_limit,
                        leakyrelu_coefficient,
                        DataTypeToEnum<T>::value,
                        &input_shape_,
                        output,
                        &kwg_size_[0],
                        op_name_,
                        elt);
    };
  } else if (kernel_h == 3 && kernel_w == 3) {
    conv_func = [&]() -> VanState {
      return Conv2dK3x3(context,
                        &kernels_[0],
                        input,
                        filter,
                        bias,
                        strides[0],
                        paddings.data(),
                        dilations,
                        activation,
                        relux_max_limit,
                        leakyrelu_coefficient,
                        DataTypeToEnum<T>::value,
                        &input_shape_,
                        output,
                        &kwg_size_[0],
                        op_name_,
                        elt);
    };
  } else {
    conv_func = [&]() -> VanState {
      return Conv2d(context,
                    &kernels_[0],
                    input,
                    filter,
                    bias,
                    strides[0],
                    paddings.data(),
                    dilations,
                    activation,
                    relux_max_limit,
                    leakyrelu_coefficient,
                    DataTypeToEnum<T>::value,
                    &input_shape_,
                    output,
                    &kwg_size_[0],
                    op_name_,
                    elt);
    };
  }

  auto result = conv_func();

  // std::vector<size_t> output_pitch = {0, 0};
  // image::WriteNHWC4DResult<float>(output, output_pitch, output->name(),
  // true);
  return result;
}

} // namespace image
} // namespace opencl
} // namespace deepvan

#endif // DEEPVAN_BACKEND_OPENCL_IMAGE_CONV_2D_H_
