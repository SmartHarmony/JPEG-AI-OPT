#include "deepvan/backend/deconv_2d_op.h"

#if defined(NEON_SUPPORT)
#include "deepvan/backend/arm/fp32/deconv_2d.h"
#include "deepvan/backend/arm/fp32/deconv_2d_general.h"
#include <arm_neon.h>
#else
#include "deepvan/backend/ref/deconv_2d.h"
#endif

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "deepvan/core/future.h"
#include "deepvan/core/tensor.h"
#include "deepvan/utils/math.h"
#include "deepvan/utils/memory.h"
#ifdef OPENCL_SUPPORT
#include "deepvan/backend/opencl/buffer_transformer.h"
#include "deepvan/backend/opencl/image/deconv_2d.h"
#endif // DEEPVAN_ENABLE_OPENCL

namespace deepvan {

template <DeviceType D, class T>
class Deconv2dOp;

template <>
class Deconv2dOp<DeviceType::CPU, float> : public Deconv2dOpBase {
public:
  explicit Deconv2dOp(OpConstructContext *context) : Deconv2dOpBase(context) {}

  VanState Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    const Tensor *filter = this->Input(1);
    const Tensor *bias = nullptr;
    const Tensor *output_shape_tensor = nullptr;
    if (model_type_ == CAFFE || model_type_ == ONNX) {
      bias = this->InputSize() >= 3 ? this->Input(2) : nullptr;
    } else {
      output_shape_tensor = this->InputSize() >= 3 ? this->Input(2) : nullptr;
      bias = this->InputSize() >= 4 ? this->Input(3) : nullptr;
    }
    Tensor *output = this->Output(0);

    CONDITIONS_NOTNULL(input);
    CONDITIONS_NOTNULL(filter);
    CONDITIONS_NOTNULL(output);

#if defined(NEON_SUPPORT)
    const index_t kernel_h = filter->dim(2);
    const index_t kernel_w = filter->dim(3);

    bool use_neon_2x2_s1 = kernel_h == kernel_w && kernel_h == 2 &&
                           strides_[0] == strides_[1] && strides_[0] == 1;
    bool use_neon_2x2_s2 = kernel_h == kernel_w && kernel_h == 2 &&
                           strides_[0] == strides_[1] && strides_[0] == 2;

    bool use_neon_3x3_s1 = kernel_h == kernel_w && kernel_h == 3 &&
                           strides_[0] == strides_[1] && strides_[0] == 1;
    bool use_neon_3x3_s2 = kernel_h == kernel_w && kernel_h == 3 &&
                           strides_[0] == strides_[1] && strides_[0] == 2;

    bool use_neon_4x4_s1 = kernel_h == kernel_w && kernel_h == 4 &&
                           strides_[0] == strides_[1] && strides_[0] == 1;
    bool use_neon_4x4_s2 = kernel_h == kernel_w && kernel_h == 4 &&
                           strides_[0] == strides_[1] && strides_[0] == 2;

    if (deconv2d_delegator_ == nullptr) {

      deconv2d_delegator_ =
          make_unique<arm::fp32::Deconv2dGeneral>(strides_,
                                                  std::vector<int>{1, 1},
                                                  paddings_,
                                                  padding_type_,
                                                  model_type_);
    }
    deconv2d_delegator_->Compute(
        context, input, filter, output_shape_tensor, output);
#else
    if (deconv2d_delegator_ == nullptr) {
      deconv2d_delegator_ =
          make_unique<ref::Deconv2d<float>>(strides_,
                                            std::vector<int>{1, 1},
                                            paddings_,
                                            padding_type_,
                                            model_type_);
    }
    deconv2d_delegator_->Compute(
        context, input, filter, output_shape_tensor, output);
#endif

    Tensor::MappingGuard bias_guard(bias);
    auto bias_data = bias == nullptr ? nullptr : bias->data<float>();
    auto output_data = output->mutable_data<float>();
    auto out_shape = output->shape();

    if (bias_data != nullptr) {
      const index_t batch = out_shape[0];
      const index_t channels = out_shape[1];
      const index_t img_size = out_shape[2] * out_shape[3];
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channels; ++c) {
          for (index_t i = 0; i < img_size; ++i) {
            output_data[(b * channels + c) * img_size + i] += bias_data[c];
          }
        }
      }
    }

    DoActivation<float>(output_data,
                        output_data,
                        output->size(),
                        activation_,
                        relux_max_limit_,
                        leakyrelu_coefficient_);

    return VanState::SUCCEED;
  }

private:
#if defined(NEON_SUPPORT)
  std::unique_ptr<arm::fp32::Deconv2dBase> deconv2d_delegator_;
#else
  std::unique_ptr<ref::Deconv2d<float>> deconv2d_delegator_;
#endif
};

#ifdef OPENCL_SUPPORT
template <typename T>
class Deconv2dOp<DeviceType::GPU, T> : public Deconv2dOpBase {
public:
  explicit Deconv2dOp(OpConstructContext *context) : Deconv2dOpBase(context) {
    MemoryType mem_type = MemoryType::GPU_IMAGE;
    if (context->device()->gpu_runtime()->UseImageMemory()) {
      kernel_ = make_unique<opencl::image::Deconv2dKernel<T>>(operator_name());
    } else {
      STUB;
    }
    CONDITIONS(TransformFilter<T>(context,
                                  operator_def_.get(),
                                  1,
                                  OpenCLBufferType::CONV2D_FILTER,
                                  mem_type) == VanState::SUCCEED);
    if (model_type_ == FrameworkType::CAFFE || model_type_ == ONNX) {
      if (operator_def_->input_size() >= 3) {
        CONDITIONS(TransformFilter<T>(context,
                                      operator_def_.get(),
                                      2,
                                      OpenCLBufferType::ARGUMENT,
                                      mem_type) == VanState::SUCCEED);
      }
    } else {
      if (operator_def_->input_size() >= 4) {
        CONDITIONS(TransformFilter<T>(context,
                                      operator_def_.get(),
                                      3,
                                      OpenCLBufferType::ARGUMENT,
                                      mem_type) == VanState::SUCCEED);
      }
      context->SetInputInfo(2, MemoryType::CPU_BUFFER, DataType::DT_INT32);
    }
  }
  VanState Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    const Tensor *filter = this->Input(1);
    const Tensor *bias = nullptr;
    const Tensor *output_shape_tensor = nullptr;
    if (model_type_ == CAFFE || model_type_ == ONNX) {
      bias = this->InputSize() >= 3 ? this->Input(2) : nullptr;
    } else {
      output_shape_tensor = this->InputSize() >= 3 ? this->Input(2) : nullptr;
      bias = this->InputSize() >= 4 ? this->Input(3) : nullptr;
    }
    Tensor *output = this->Output(0);

    CONDITIONS_NOTNULL(input);
    CONDITIONS_NOTNULL(filter);
    CONDITIONS_NOTNULL(output);

    std::vector<int> in_paddings(2, 0);
    std::vector<index_t> out_shape(4, 0);

    if (model_type_ == FrameworkType::TENSORFLOW) {
      CONDITIONS_NOTNULL(output_shape_tensor);
      CONDITIONS(output_shape_tensor->size() == 4);
      Tensor::MappingGuard output_shape_mapper(output_shape_tensor);
      auto output_shape_data = output_shape_tensor->data<int32_t>();
      out_shape =
          std::vector<index_t>(output_shape_data, output_shape_data + 4);

      CalcDeconvShape_TF(input->shape().data(),
                         filter->shape().data(),
                         out_shape.data(),
                         strides_.data(),
                         1,
                         padding_type_,
                         in_paddings.data(),
                         nullptr,
                         nullptr);
    } else {
      std::vector<int> out_paddings(2, 0);
      if (!paddings_.empty())
        out_paddings = paddings_;
      CalcDeconvShape_Caffe(input->shape().data(),
                            filter->shape().data(),
                            strides_.data(),
                            out_paddings.data(),
                            output_padding_.data(),
                            1,
                            in_paddings.data(),
                            out_shape.data(),
                            nullptr);
    }

    return kernel_->Compute(context,
                            input,
                            filter,
                            bias,
                            strides_.data(),
                            in_paddings.data(),
                            activation_,
                            relux_max_limit_,
                            leakyrelu_coefficient_,
                            out_shape,
                            output);
  }

private:
  std::unique_ptr<OpenCLDeconv2dKernel> kernel_;
};
#endif // DEEPVAN_ENABLE_OPENCL

void RegisterDeconv2D(OpRegistryBase *op_registry) {
  VAN_REGISTER_OP(op_registry, "Deconv2D", Deconv2dOp, DeviceType::CPU, float);
#ifdef OPENCL_SUPPORT
  VAN_REGISTER_OP(op_registry, "Deconv2D", Deconv2dOp, DeviceType::GPU, float);
  VAN_REGISTER_OP(op_registry, "Deconv2D", Deconv2dOp, DeviceType::GPU, half);
#endif // DEEPVAN_ENABLE_OPENCL
  op_registry->Register(
      OpConditionBuilder("Deconv2D")
          .SetDevicePlacerFunc(
              [](OpConstructContext *context) -> std::set<DeviceType> {
                std::set<DeviceType> result;
                auto op_def = context->operator_def();
                auto executing_on =
                    ProtoArgHelper::GetRepeatedArgs<OperatorProto, int>(
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
