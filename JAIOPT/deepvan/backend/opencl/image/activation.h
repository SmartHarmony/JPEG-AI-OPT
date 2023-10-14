#ifndef DEEPVAN_BACKEND_OPENCL_IMAGE_ACTIVATION_H_
#define DEEPVAN_BACKEND_OPENCL_IMAGE_ACTIVATION_H_

#include "deepvan/backend/opencl/activation.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "deepvan/backend/common/activation_type.h"
#include "deepvan/backend/opencl/helper.h"
#include "deepvan/core/op_context.h"
#include "deepvan/core/tensor.h"

namespace deepvan {
namespace opencl {
namespace image {

template <typename T>
class ActivationKernel : public OpenCLActivationKernel {
public:
  ActivationKernel(ActivationType type,
                   T relux_max_limit,
                   T leakyrelu_coefficient)
      : activation_(type),
        relux_max_limit_(relux_max_limit),
        leakyrelu_coefficient_(leakyrelu_coefficient) {}

  VanState Compute(OpContext *context,
                   const Tensor *input,
                   const Tensor *alpha,
                   Tensor *output) override;

private:
  VanState Compute2DTensor(OpContext *context,
                           const Tensor *input,
                           const Tensor *alpha,
                           Tensor *output);

  VanState Compute4DTensor(OpContext *context,
                           const Tensor *input,
                           const Tensor *alpha,
                           Tensor *output);

  VanState ComputeBertTensor(OpContext *context,
                             const Tensor *input,
                             const Tensor *alpha,
                             Tensor *output);

  ActivationType activation_;
  T relux_max_limit_;
  T leakyrelu_coefficient_;
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
  std::string tuning_key_prefix_;
};

template <typename T>
VanState ActivationKernel<T>::Compute(OpContext *context,
                                      const Tensor *input,
                                      const Tensor *alpha,
                                      Tensor *output) {
  if (input->dim_size() == 4) {
    return Compute4DTensor(context, input, alpha, output);
  } else if (input->dim_size() == 2) {
    return Compute2DTensor(context, input, alpha, output);
  } else {
    //todo: not safe.
    return ComputeBertTensor(context, input, alpha, output);
  }
  return VanState::UNSUPPORTED;
}

template <typename T>
VanState ActivationKernel<T>::Compute2DTensor(OpContext *context,
                                              const Tensor *input,
                                              const Tensor *alpha,
                                              Tensor *output) {
  auto output_image_shape = output->image_shape();
  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG;
    NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = DEEPVAN_OBFUSCATE_SYMBOL("activation2d_tensor");
    built_options.emplace("-Dactivation2d_tensor=" + kernel_name);
    auto dt = DataTypeToEnum<T>::value;
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
    switch (activation_) {
    case RELU: built_options.emplace("-DUSE_RELU"); break;
    case RELUX: built_options.emplace("-DUSE_RELUX"); break;
    case PRELU: built_options.emplace("-DUSE_PRELU"); break;
    case TANH: built_options.emplace("-DUSE_TANH"); break;
    case SIGMOID: built_options.emplace("-DUSE_SIGMOID"); break;
    case LEAKYRELU: built_options.emplace("-DUSE_LEAKYRELU"); break;
    default: LOG(FATAL) << "Unknown activation type: " << activation_;
    }
    RETURN_IF_ERROR(runtime->BuildKernel(
        "activation", kernel_name, built_options, &kernel_));
  }

  const uint32_t gws[2] = {static_cast<uint32_t>(output_image_shape[0]),
                           static_cast<uint32_t>(output_image_shape[1])};
  VLOG(INFO) << DEBUG_GPU
             << "Activation output shape: " << MakeString(output->shape())
             << ", output image shape: " << MakeString(output_image_shape)
             << ", gws: [" << gws[0] << ", " << gws[1] << "]";

  OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    int idx = 0;
    OUT_OF_RANGE_SET_ARGS(kernel_);
    SET_2D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_image()));
    if (activation_ == PRELU) {
      CONDITIONS_NOTNULL(alpha);
      kernel_.setArg(idx++, *(alpha->opencl_image()));
    }
    kernel_.setArg(idx++, static_cast<float>(relux_max_limit_));
    kernel_.setArg(idx++, static_cast<float>(leakyrelu_coefficient_));
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = input->shape();
  }

  const std::vector<uint32_t> lws = {32, 1};
  RETURN_IF_ERROR(Run2DKernel(runtime, kernel_, gws, lws, context->future()));
  OUT_OF_RANGE_VALIDATION;
  return VanState::SUCCEED;
}

template <typename T>
VanState ActivationKernel<T>::Compute4DTensor(OpContext *context,
                                              const Tensor *input,
                                              const Tensor *alpha,
                                              Tensor *output) {
  const index_t batch = input->dim(0);
  const index_t height = input->dim(1);
  const index_t width = input->dim(2);
  const index_t channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG;
    NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = DEEPVAN_OBFUSCATE_SYMBOL("activation4d_tensor");
    built_options.emplace("-Dactivation4d_tensor=" + kernel_name);
    auto dt = DataTypeToEnum<T>::value;
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
    switch (activation_) {
    case RELU:
      tuning_key_prefix_ = "relu_opencl_kernel";
      built_options.emplace("-DUSE_RELU");
      break;
    case RELUX:
      tuning_key_prefix_ = "relux_opencl_kernel";
      built_options.emplace("-DUSE_RELUX");
      break;
    case PRELU:
      tuning_key_prefix_ = "prelu_opencl_kernel";
      built_options.emplace("-DUSE_PRELU");
      break;
    case TANH:
      tuning_key_prefix_ = "tanh_opencl_kernel";
      built_options.emplace("-DUSE_TANH");
      break;
    case SIGMOID:
      tuning_key_prefix_ = "sigmoid_opencl_kernel";
      built_options.emplace("-DUSE_SIGMOID");
      break;
    case LEAKYRELU:
      tuning_key_prefix_ = "leakyrelu_opencl_kernel";
      built_options.emplace("-DUSE_LEAKYRELU");
      break;
    case HARDSIGMOID:
      tuning_key_prefix_ = "hard_sigmoid_kernel";
      built_options.emplace("-DUSE_HARDSIGMOID");
      break;
    case COS:
      tuning_key_prefix_ = "cos_kernel";
      built_options.emplace("-DUSE_COS");
      break;
    default: LOG(FATAL) << "Unknown activation type: " << activation_;
    }
    RETURN_IF_ERROR(runtime->BuildKernel(
        "activation", kernel_name, built_options, &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};
  OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    int idx = 0;
    OUT_OF_RANGE_SET_ARGS(kernel_);
    SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_image()));
    if (activation_ == PRELU) {
      CONDITIONS_NOTNULL(alpha);
      kernel_.setArg(idx++, *(alpha->opencl_image()));
    }
    kernel_.setArg(idx++, static_cast<float>(relux_max_limit_));
    kernel_.setArg(idx++, static_cast<float>(leakyrelu_coefficient_));
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = input->shape();
  }

  const std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_);
  RETURN_IF_ERROR(Run3DKernel(runtime, kernel_, gws, lws, context->future()));

  OUT_OF_RANGE_VALIDATION;
  return VanState::SUCCEED;
}


// Remove this! 
template <typename T>
VanState ActivationKernel<T>::ComputeBertTensor(OpContext *context,
                                                const Tensor *input,
                                                const Tensor *alpha,
                                                Tensor *output) {
  auto output_image_shape = output->image_shape();
  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG;
    NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = DEEPVAN_OBFUSCATE_SYMBOL("activation_bert");
    built_options.emplace("-Dactivation_bert=" + kernel_name);
    auto dt = DataTypeToEnum<T>::value;
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
    switch (activation_) {
    case RELU: built_options.emplace("-DUSE_RELU"); break;
    case RELUX: built_options.emplace("-DUSE_RELUX"); break;
    case PRELU: built_options.emplace("-DUSE_PRELU"); break;
    case TANH: built_options.emplace("-DUSE_TANH"); break;
    case SIGMOID: built_options.emplace("-DUSE_SIGMOID"); break;
    case LEAKYRELU: built_options.emplace("-DUSE_LEAKYRELU"); break;
    case COS: built_options.emplace("-DUSE_COS"); break;
    default: LOG(FATAL) << "Unknown activation type: " << activation_;
    }
    RETURN_IF_ERROR(runtime->BuildKernel(
        "activation", kernel_name, built_options, &kernel_));
    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  const uint32_t gws[2] = {static_cast<uint32_t>(output_image_shape[0]),
                           static_cast<uint32_t>(output_image_shape[1])};

  OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    int idx = 0;
    OUT_OF_RANGE_SET_ARGS(kernel_);
    SET_2D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_image()));
    if (activation_ == PRELU) {
      CONDITIONS_NOTNULL(alpha);
      kernel_.setArg(idx++, *(alpha->opencl_image()));
    }
    kernel_.setArg(idx++, static_cast<float>(relux_max_limit_));
    kernel_.setArg(idx++, static_cast<float>(leakyrelu_coefficient_));
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = input->shape();
  }

  const std::vector<uint32_t> lws = Default2DLocalWS(runtime, gws, kwg_size_);
  RETURN_IF_ERROR(Run2DKernel(runtime, kernel_, gws, lws, context->future()));
  OUT_OF_RANGE_VALIDATION;
  return VanState::SUCCEED;
}

} // namespace image
} // namespace opencl
} // namespace deepvan

#endif // DEEPVAN_BACKEND_OPENCL_IMAGE_ACTIVATION_H_
