#include <memory>

#include "deepvan/backend/opencl/buffer_transformer.h"
// #include "deepvan/backend/opencl/buffer_transformer_c3d.h"
#include "deepvan/core/operator.h"

namespace deepvan {

template <DeviceType D, class T>
class BufferTransformOp;

template <typename T>
class BufferTransformOp<DeviceType::GPU, T> : public Operation {
public:
  explicit BufferTransformOp(OpConstructContext *context)
      : Operation(context),
        wino_blk_size_(Operation::GetOptionalArg<int>("wino_block_size", 0)),
        out_mem_type_(static_cast<MemoryType>(Operation::GetOptionalArg<int>(
            "mem_type",
            static_cast<int>(MemoryType::GPU_IMAGE)))) {}

  VanState Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    auto type = static_cast<OpenCLBufferType>(Operation::GetOptionalArg<int>(
        "buffer_type", static_cast<int>(CONV2D_FILTER)));
    bool has_data_format =
        Operation::GetOptionalArg<int>("has_data_format", 0) != 0;
    MemoryType in_mem_type =
        context->workspace()->GetTensor(operator_def_->input(0))->memory_type();

    VLOG(INFO) << DEBUG_GPU << "Executing Buffer transform op "
               << ", input shape: " << MakeString(input->shape())
               << ", output shape: " << MakeString(output->shape())
               << ", has_data_format: " << has_data_format
               << ", in_mem_type: " << in_mem_type
               << ", out_mem_type_: " << out_mem_type_
               << ", pruning_type_: " << pruning_type_
               << ", model_type: " << model_type_;

    return OpenCLBufferTransformer<T>(
               in_mem_type, out_mem_type_, pruning_type_, model_type_)
        .Transform(context,
                   input,
                   type,
                   out_mem_type_,
                   wino_blk_size_,
                   has_data_format,
                   model_type_,
                   output);
  }

private:
  const int wino_blk_size_;
  MemoryType out_mem_type_;
};

void RegisterBufferTransform(OpRegistryBase *op_registry) {
  VAN_REGISTER_OP(op_registry,
                  "BufferTransform",
                  BufferTransformOp,
                  DeviceType::GPU,
                  float);

  VAN_REGISTER_OP(
      op_registry, "BufferTransform", BufferTransformOp, DeviceType::GPU, half);
}

} // namespace deepvan
