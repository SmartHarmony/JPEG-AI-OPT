#include <algorithm>
#include <arm_neon.h>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "deepvan/backend/activation_op.h"
#include "deepvan/backend/arm/common/tensor_logger.h"
#include "deepvan/backend/arm/fp32/conv_2d.h"
#include "deepvan/backend/arm/fp32/conv_2d_1x1.h"
#include "deepvan/backend/arm/fp32/conv_2d_kernel.h"
#include "deepvan/backend/arm/fp32/conv_gemm.h"
#include "deepvan/backend/arm/fp32/conv_general.h"
#include "deepvan/backend/common/conv_pool_2d_util.h"
#include "deepvan/backend/common/im2col.h"
#include "deepvan/backend/conv_pool_2d_op_base.h"
#include "deepvan/backend/opencl/buffer_transformer.h"
#include "deepvan/backend/opencl/image/conv_2d.h"
#include "deepvan/core/future.h"
#include "deepvan/core/operator.h"
#include "deepvan/core/tensor.h"
#include "deepvan/utils/math.h"
#include "deepvan/utils/memory.h"

namespace deepvan {

template <DeviceType D, class T>
class Conv2dOp;

template <>
class Conv2dOp<DeviceType::CPU, float> : public ConvPool2dOpBase {
public:
  explicit Conv2dOp(OpConstructContext *context)
      : ConvPool2dOpBase(context),
        activation_(StringToActivationType(
            Operation::GetOptionalArg<std::string>("activation", "NOOP"))),
        relux_max_limit_(Operation::GetOptionalArg<float>("max_limit", 0.0f)),
        leakyrelu_coefficient_(
            Operation::GetOptionalArg<float>("leakyrelu_coefficient", 0.0f)) {}

  VanState Run(OpContext *context) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);

    const index_t channels = filter->dim(0);

#ifdef NEON_SUPPORT
    // the following params are used to decide which conv delegator to use
    const index_t stride_h = strides_[0];
    const index_t stride_w = strides_[1];
    const index_t dilation_h = dilations_[0];
    const index_t dilation_w = dilations_[1];
    const index_t filter_h = filter->dim(2);
    const index_t filter_w = filter->dim(3);
    const index_t input_channels = input->dim(1);

    if (conv2d_common_delegator_.get() == nullptr) {
      const std::vector<int> kernels = {(int)filter_h, (int)filter_w};
      if (pruning_type_ == PruningType::DENSE) {
        if (false && filter_h == 3 && filter_w == 3 && dilation_h == 1 &&
            dilation_w == 1 && stride_h == 1 && stride_w == 1) {
          conv2d_common_delegator_ = make_unique<arm::fp32::Conv2dGEMM>(
              strides_, dilations_, paddings_, padding_type_);
        } else if (filter_h == 1 && filter_w == 1 && dilation_h == 1 &&
                   dilation_w == 1 && stride_h == 1 && stride_w == 1) {
          conv2d_common_delegator_ = make_unique<arm::fp32::Conv2dK1x1>(
              paddings_, padding_type_, strides_);
        } else if (filter_h == 1 && filter_w == 1 && dilation_h == 1 &&
                   dilation_w == 1 && stride_h == 2 && stride_w == 2) {
          conv2d_common_delegator_ = make_unique<arm::fp32::Conv2dK1x1>(
              paddings_, padding_type_, strides_);
        } else {
          conv2d_common_delegator_ = make_unique<arm::fp32::Conv2dGeneral>(
              strides_, dilations_, paddings_, padding_type_);
        }
      } // DENSE
    }   // end nullptr

    CONDITIONS(conv2d_common_delegator_ != nullptr, "delegator cannot be null");
    conv2d_common_delegator_->Compute(context, input, filter, output);
#else
    if (ref_conv2d_delegator_.get() == nullptr) {
      ref_conv2d_delegator_ = make_unique<ref::Conv2d<float>>(
          strides_, dilations_, paddings_, padding_type_);
    }
    ref_conv2d_delegator_->Compute(context, input, filter, output);
#endif

    // add bias
    Tensor::MappingGuard bias_guard(bias);
    Tensor::MappingGuard output_guard(output);
    auto bias_data = bias == nullptr ? nullptr : bias->data<float>();
    auto output_data = output->mutable_data<float>();
    if (bias_data != nullptr) {

      const index_t batch = input->dim(0);
      const index_t height = output->dim(2);
      const index_t width = output->dim(3);
      const index_t image_size = height * width;
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channels; ++c) {
          float *output_ptr = output_data + (b * channels + c) * image_size;
          const float bias = bias_data[c];
#if defined(NEON_SUPPORT)
          float32x4_t vbias = vdupq_n_f32(bias);
          for (index_t i = 0; i <= image_size - 4; i += 4) {
            float32x4_t v = vld1q_f32(output_ptr + i);
            v = vaddq_f32(v, vbias);
            vst1q_f32(output_ptr + i, v);
          }
          for (index_t i = (image_size >> 2) << 2; i < image_size; ++i) {
            output_ptr[i] += bias;
          }
#else
          for (index_t i = 0; i < image_size; ++i) {
            output_ptr[i] += bias;
          }
#endif
        }
      }
      // }
    }

    DoActivation(output_data,
                 output_data,
                 output->size(),
                 activation_,
                 relux_max_limit_,
                 leakyrelu_coefficient_);

    // debugger::WriteTensor2File(output, operator_name());
    return VanState::SUCCEED;
  }

private:
  const ActivationType activation_;
  const float relux_max_limit_;
  const float leakyrelu_coefficient_;
#ifdef NEON_SUPPORT
  std::unique_ptr<arm::Conv2dCommon> conv2d_common_delegator_;
#else
  std::unique_ptr<ref::Conv2d<float>> ref_conv2d_delegator_;
#endif // NEON_SUPPORT

private:
  DEEPVAN_OP_INPUT_TAGS(INPUT, FILTER, BIAS);
  DEEPVAN_OP_OUTPUT_TAGS(OUTPUT);
};

#ifdef OPENCL_SUPPORT
template <typename T>
class Conv2dOp<DeviceType::GPU, T> : public ConvPool2dOpBase {
public:
  explicit Conv2dOp(OpConstructContext *context)
      : ConvPool2dOpBase(context),
        activation_(StringToActivationType(
            Operation::GetOptionalArg<std::string>("activation", "NOOP"))),
        comp_type_(Operation::GetOptionalArg<std::string>("comp_type", "CONV")),
        relux_max_limit_(Operation::GetOptionalArg<float>("max_limit", 0.0f)),
        leakyrelu_coefficient_(
            Operation::GetOptionalArg<float>("leakyrelu_coefficient", 0.0f)),
        wino_block_size_(Operation::GetOptionalArg<int>("wino_block_size", 0)) {

    // DENSE and PATTERN
    MemoryType mem_type;

    bool gomix = false;
    auto filter_shape =
        context->workspace()->GetTensor(operator_def_->input(1))->shape();
    bool mali =
        context->device()->gpu_runtime()->opencl_runtime()->gpu_type() ==
        GPUType::MALI;
    if (filter_shape[2] == 3 && filter_shape[3] == 3 && mali &&
        wino_block_size_ == 0) {
      gomix = true;
    } else {
      gomix = false;
    }
    gomix = false; // 31 -> 27

    if (context->device()->gpu_runtime()->UseImageMemory()) {
      mem_type = MemoryType::GPU_IMAGE;
      kernel_ = make_unique<opencl::image::Conv2dKernel<T>>(operator_name());
    }
    context->set_output_mem_type(mem_type);
    // Transform filter tensor to target format
    if (context->device()->gpu_runtime()->UseImageMemory() &&
        (wino_block_size_ == 2 || wino_block_size_ == 4) &&
        (kernel_->CheckUseWinograd(
            context->device()->gpu_runtime()->opencl_runtime(),
            context->workspace()->GetTensor(operator_def_->input(1))->shape(),
            std::vector<index_t>(operator_def_->output_shape(0).dims().begin(),
                                 operator_def_->output_shape(0).dims().end()),
            strides_.data(),
            dilations_.data(),
            &wino_block_size_))) {
      CONDITIONS(TransformFilter<T>(context,
                                    operator_def_.get(),
                                    1,
                                    OpenCLBufferType::WINOGRAD_FILTER,
                                    mem_type,
                                    wino_block_size_) == VanState::SUCCEED);
    } else { // wino
      LOG(WARNING) << "Use non-Winograd version of Convolution with filter: "
                   << MakeString(context->workspace()
                                     ->GetTensor(operator_def_->input(1))
                                     ->shape());
      wino_block_size_ = 0;
      if (gomix) {
        CONDITIONS(TransformFilter<T>(context,
                                      operator_def_.get(),
                                      1,
                                      OpenCLBufferType::CONV2D_FILTER_BUFFER,
                                      MemoryType::GPU_BUFFER) ==
                   VanState::SUCCEED);
      } else { // MALI
        CONDITIONS(TransformFilter<T>(context,
                                      operator_def_.get(),
                                      1,
                                      mem_type == MemoryType::GPU_IMAGE
                                          ? OpenCLBufferType::CONV2D_FILTER
                                          : OpenCLBufferType::BUFFER_2_BUFFER,
                                      mem_type) == VanState::SUCCEED);
      }
    } // end wino
    if (operator_def_->input_size() > 2) {
      if (gomix) {
        CONDITIONS(TransformFilter<T>(context,
                                      operator_def_.get(),
                                      2,
                                      OpenCLBufferType::ARGUMENT,
                                      MemoryType::GPU_BUFFER) ==
                   VanState::SUCCEED);
      } else {
        CONDITIONS(TransformFilter<T>(context,
                                      operator_def_.get(),
                                      2,
                                      OpenCLBufferType::ARGUMENT,
                                      mem_type) == VanState::SUCCEED);
      }
    }
  }
  VanState Run(OpContext *context) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    const Tensor *elt = this->InputSize() >= 4 ? Input(ELEMENT) : nullptr;
    Tensor *output = this->Output(OUTPUT);

    const index_t stride_h = strides_[0];
    const index_t stride_w = strides_[1];
    const index_t dilation_h = dilations_[0];
    const index_t dilation_w = dilations_[1];
    const index_t filter_h = filter->dim(2);
    const index_t filter_w = filter->dim(3);

    return kernel_->Compute(context,
                            input,
                            filter,
                            bias,
                            strides_.data(),
                            padding_type_,
                            paddings_,
                            dilations_.data(),
                            activation_,
                            relux_max_limit_,
                            leakyrelu_coefficient_,
                            wino_block_size_,
                            output,
                            elt);
  }

private:
  const ActivationType activation_;
  const float relux_max_limit_;
  const float leakyrelu_coefficient_;
  std::unique_ptr<OpenCLConv2dKernel> kernel_;

  int wino_block_size_;
  std::string comp_type_;

private:
  DEEPVAN_OP_INPUT_TAGS(INPUT, FILTER, BIAS, ELEMENT);
  DEEPVAN_OP_OUTPUT_TAGS(OUTPUT);
};
#endif // OPENCL_SUPPORT

void RegisterConv2D(OpRegistryBase *op_registry) {
  VAN_REGISTER_OP(op_registry, "Conv2D", Conv2dOp, DeviceType::CPU, float);

#ifdef OPENCL_SUPPORT
  VAN_REGISTER_OP(op_registry, "Conv2D", Conv2dOp, DeviceType::GPU, float);
  VAN_REGISTER_OP(op_registry, "Conv2D", Conv2dOp, DeviceType::GPU, half);
#endif // OPENCL_SUPPORT
  op_registry->Register(OpConditionBuilder("Conv2D").SetDevicePlacerFunc(
      [](OpConstructContext *context) -> std::set<DeviceType> {
        std::set<DeviceType> result;
        auto op_def = context->operator_def();
        auto executing_on = ProtoArgHelper::GetRepeatedArgs<OperatorProto, int>(
            *op_def,
            "executing_on",
            {static_cast<int>(DeviceType::CPU),
             static_cast<int>(DeviceType::GPU)});
        for (auto exe : executing_on) {
          result.insert(static_cast<DeviceType>(exe));
        }
        return result;
      }));
}

} // namespace deepvan
