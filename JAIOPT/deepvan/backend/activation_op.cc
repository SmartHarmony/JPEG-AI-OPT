#include "deepvan/backend/activation_op.h"

#include <memory>

#include "deepvan/core/operator.h"

#ifdef OPENCL_SUPPORT
#include "deepvan/backend/opencl/buffer_transformer.h"
#include "deepvan/backend/opencl/image/activation.h"
#endif // OPENCL_SUPPORT
#include "deepvan/utils/memory.h"

namespace deepvan {

template <DeviceType D, class T>
class ActivationOp;

template <>
class ActivationOp<DeviceType::CPU, float> : public Operation {
public:
  explicit ActivationOp(OpConstructContext *context)
      : Operation(context),
        activation_(StringToActivationType(
            Operation::GetOptionalArg<std::string>("activation", "NOOP"))),
        relux_max_limit_(Operation::GetOptionalArg<float>("max_limit", 0.0f)),
        leakyrelu_coefficient_(
            Operation::GetOptionalArg<float>("leakyrelu_coefficient", 0.0f)) {}

  VanState Run(OpContext *context) override {
    UNUSED_VARIABLE(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    RETURN_IF_ERROR(output->ResizeLike(input));

    const float *input_ptr = input->data<float>();
    float *output_ptr = output->mutable_data<float>();
    if (activation_ == PRELU) {
      CONDITIONS(this->InputSize() > 1);
      const Tensor *alpha = this->Input(1);
      const float *alpha_ptr = alpha->data<float>();
      const index_t outer_size = output->dim(0);
      const index_t inner_size = output->dim(2) * output->dim(3);
      PReLUActivation(input_ptr,
                      outer_size,
                      input->dim(1),
                      inner_size,
                      alpha_ptr,
                      output_ptr);
    } else {
      DoActivation(input_ptr,
                   output_ptr,
                   output->size(),
                   activation_,
                   relux_max_limit_,
                   leakyrelu_coefficient_);
    }
    return VanState::SUCCEED;
  }

private:
  ActivationType activation_;
  float relux_max_limit_;
  float leakyrelu_coefficient_;
};

#ifdef OPENCL_SUPPORT
template <typename T>
class ActivationOp<DeviceType::GPU, T> : public Operation {
public:
  explicit ActivationOp(OpConstructContext *context) : Operation(context) {
    ActivationType type = StringToActivationType(
        Operation::GetOptionalArg<std::string>("activation", "NOOP"));
    auto relux_max_limit =
        static_cast<T>(Operation::GetOptionalArg<float>("max_limit", 0.0f));
    auto leakyrelu_coefficient = static_cast<T>(
        Operation::GetOptionalArg<float>("leakyrelu_coefficient", 0.0f));
    MemoryType mem_type;
    if (context->device()->gpu_runtime()->UseImageMemory()) {
      context->set_output_mem_type(MemoryType::GPU_IMAGE);
      kernel_ = make_unique<opencl::image::ActivationKernel<T>>(
          type, relux_max_limit, leakyrelu_coefficient);
    }
    if (type == ActivationType::PRELU) {
      CONDITIONS(TransformFilter<T>(context,
                                    operator_def_.get(),
                                    1,
                                    OpenCLBufferType::ARGUMENT,
                                    mem_type,
                                    0,
                                    pruning_type_) == VanState::SUCCEED);
    }
  }

  VanState Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    const Tensor *alpha = this->InputSize() > 1 ? this->Input(1) : nullptr;
    Tensor *output = this->Output(0);
    RETURN_IF_ERROR(output->ResizeLike(input));
    return kernel_->Compute(context, input, alpha, output);
  }

private:
  std::unique_ptr<OpenCLActivationKernel> kernel_;
};
#endif // OPENCL_SUPPORT

void RegisterActivation(OpRegistryBase *op_registry) {
  VAN_REGISTER_OP(
      op_registry, "Activation", ActivationOp, DeviceType::CPU, float);

#ifdef OPENCL_SUPPORT
  VAN_REGISTER_OP(
      op_registry, "Activation", ActivationOp, DeviceType::GPU, float);

  VAN_REGISTER_OP(
      op_registry, "Activation", ActivationOp, DeviceType::GPU, half);
#endif // OPENCL_SUPPORT

  // VAN_REGISTER_OP_CONDITION(
  //   op_registry,
  //   OpConditionBuilder("Activation")
  //       .SetDevicePlacerFunc(
  //           [](OpConstructContext *context) -> std::set<DeviceType> {
  //             UNUSED_VARIABLE(context);
  //             return { DeviceType::CPU, DeviceType::GPU };
  //           }));
  op_registry->Register(
      OpConditionBuilder("Activation")
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
