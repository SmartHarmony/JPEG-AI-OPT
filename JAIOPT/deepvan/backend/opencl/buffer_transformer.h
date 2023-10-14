#ifndef DEEPVAN_BACKEND_OPENCL_BUFFER_TRANSFORMER_H_
#define DEEPVAN_BACKEND_OPENCL_BUFFER_TRANSFORMER_H_

#include <memory>
#include <string>
#include <vector>

#include "deepvan/backend/common/transpose.h"
// #include "deepvan/backend/opencl/buffer/buffer_transform.h"
// #include "deepvan/backend/opencl/column/buffer_to_image.h"
// #include "deepvan/backend/opencl/column/buffer_transform.h"
// #include "deepvan/backend/opencl/csr/buffer_to_image.h"
// #include "deepvan/backend/opencl/csr/buffer_transform.h"
#include "deepvan/backend/opencl/image/buffer_to_image.h"
// #include "deepvan/backend/opencl/image/buffer_to_image_bert.h"
#include "deepvan/backend/opencl/image/image_to_buffer.h"
// #include "deepvan/backend/opencl/image/image_to_buffer_bert.h"
// #include "deepvan/backend/opencl/pattern/buffer_to_image.h"
// #include "deepvan/backend/opencl/pattern/image_to_buffer.h"
// #include "deepvan/backend/opencl/slice/buffer_to_image.h"
// #include "deepvan/backend/opencl/slice/image_to_buffer.h"
#include "deepvan/core/operator.h"
#include "deepvan/utils/memory.h"

namespace deepvan {
// Only used for GPU Operation(BufferTransform)
template <typename T>
class OpenCLBufferTransformer {
public:
  OpenCLBufferTransformer(
      const MemoryType in_mem_type,
      const MemoryType out_mem_type,
      PruningType pruning_type = PruningType::DENSE,
      ModelType model_type = ModelType::DEFAULT,
      OpenCLBufferType type = OpenCLBufferType::CONV2D_FILTER_BUFFER)
      : pruning_type_(pruning_type) {

    switch (pruning_type) {
    case PruningType::DENSE:
      UNUSED_VARIABLE(type);
      if (out_mem_type == MemoryType::GPU_IMAGE) {
        // transform the input from GPU Buffer to GPU Image

        if (model_type != ModelType::BERT) {
          kernel_ = make_unique<opencl::image::BufferToImage<T>>();
        }
      } else if (in_mem_type == MemoryType::GPU_IMAGE) {
        // transform the output from GPU Image to CPU Buffer

        { kernel_ = make_unique<opencl::image::ImageToBuffer<T>>(); }
      } else {
        // transform the buffer to buffer

        { kernel_ = make_unique<opencl::image::BufferToImage<T>>(); }
      }
      break;

    default:
      CONDITIONS(false,
                 "Deepvan does not support such kind of pruning type now");
    }
  }

  VanState Transform(OpContext *context,
                     const Tensor *input,
                     const OpenCLBufferType type,
                     const MemoryType out_mem_type,
                     const int wino_blk_size,
                     bool has_data_format,
                     ModelType model_type,
                     Tensor *output) {

    return TransformDenseAndPattern(context,
                                    input,
                                    type,
                                    out_mem_type,
                                    wino_blk_size,
                                    has_data_format,
                                    model_type,
                                    output);
  }

  VanState TransformColumn(OpContext *context,
                           const Tensor *input,
                           const OpenCLBufferType type,
                           const MemoryType out_mem_type,
                           bool has_data_format,
                           Tensor *output) {
    UNUSED_VARIABLE(has_data_format);
    NetworkController *ws = context->workspace();
    DataType dt = DataTypeToEnum<T>::value;
    MemoryType in_mem_type = input->memory_type();
    if (in_mem_type == MemoryType::GPU_BUFFER &&
        (out_mem_type == MemoryType::GPU_BUFFER ||
         out_mem_type == MemoryType::GPU_IMAGE)) { // args, filter
      return kernel_->Compute(context, input, type, 0, output);
    } else if (in_mem_type == MemoryType::CPU_BUFFER &&
               out_mem_type == MemoryType::GPU_BUFFER) { // input
      if (input->data_format() != output->data_format()) {
        // Store input into intermediate GPU tensor then convert to output
        Tensor *internal_tensor =
            ws->CreateTensor(InternalTransformedName(input->name()),
                             context->device()->allocator(),
                             input->dtype());
        internal_tensor->Resize(input->shape());
        internal_tensor->set_data_format(DataFormat::NCHW);
        Tensor::MappingGuard guard(internal_tensor);
        float *internal_ptr = internal_tensor->mutable_data<float>();
        const float *input_ptr = input->data<float>();
        memcpy(internal_ptr, input_ptr, input->raw_size());
        kernel_->Compute(context, internal_tensor, type, 0, output);
      } else {
        // Copy data from input to output
        VLOG(2) << "Transform CPU Buffer " << input->name() << " to GPU Buffer "
                << output->name() << " with data type " << dt;
        output->Resize(input->shape());
        const uint8_t *input_ptr = input->data<uint8_t>();
        Tensor::MappingGuard guard(output);
        uint8_t *output_ptr = output->mutable_data<uint8_t>();
        memcpy(output_ptr, input_ptr, input->raw_size());
      }
    } else if (in_mem_type == MemoryType::GPU_BUFFER &&
               out_mem_type == MemoryType::CPU_BUFFER) { // output
      if (input->data_format() != output->data_format()) {
        // convert to intermediate first, then copy to output
        Tensor internal_tensor(context->device()->allocator(),
                               dt,
                               false,
                               InternalTransformedName(input->name()));
        kernel_->Compute(context, input, type, 0, &internal_tensor);
        output->set_data_format(DataFormat::NCHW);
        Tensor::MappingGuard guard(&internal_tensor);
        const float *internal_ptr = internal_tensor.data<float>();
        output->Resize(internal_tensor.shape());
        float *output_ptr = output->mutable_data<float>();
        memcpy(output_ptr, internal_ptr, internal_tensor.raw_size());
      } else {
        VLOG(2) << "Transform GPU Buffer " << input->name() << " to CPU Buffer "
                << output->name() << " with data type " << dt;
        Tensor::MappingGuard guard(output);
        const T *input_ptr = input->data<T>();
        output->Resize(input->shape());
        T *output_ptr = output->mutable_data<T>();
        memcpy(output_ptr, input_ptr, input->size() * sizeof(T));
      }
    } else {
      LOG(FATAL) << "Unexpected error: " << out_mem_type;
    }
    return VanState::SUCCEED;
  }

  VanState TransformDenseAndPattern(OpContext *context,
                                    const Tensor *input,
                                    const OpenCLBufferType type,
                                    const MemoryType out_mem_type,
                                    const int wino_blk_size,
                                    bool has_data_format,
                                    ModelType model_type,
                                    Tensor *output) {
    NetworkController *ws = context->workspace();
    DataType dt = DataTypeToEnum<T>::value;
    MemoryType in_mem_type = input->memory_type();

    VLOG(INFO) << DEBUG_GPU << "Fucked Transform "
               << ", in memory type: " << in_mem_type
               << ", out memory type: " << out_mem_type
               << ", input name: " << input->name()
               << ", input shape: " << MakeString(input->shape())
               << ", output name: " << output->name()
               << ", has_data_format: " << has_data_format
               << ", opencl buffer: " << type;
    if (out_mem_type == MemoryType::GPU_IMAGE ||
        out_mem_type == MemoryType::GPU_BUFFER) {
      if (in_mem_type != MemoryType::CPU_BUFFER) {
        return kernel_->Compute(context, input, type, wino_blk_size, output);
      } else {
        // convert to the GPU Buffer with the input's data type.
        // 1. CPU buffer to GPU Buffer
        Tensor *internal_tensor =
            ws->CreateTensor(InternalTransformedName(input->name()),
                             context->device()->allocator(),
                             input->dtype());
        VLOG(INFO) << "Transform CPU Buffer " << input->name()
                   << " to GPU Buffer " << internal_tensor->name()
                   << " with data type " << dt;
        if (has_data_format && input->shape().size() == 4 &&
            model_type != ModelType::BERT) {
          // 1. (NCHW -> NHWC)
          std::vector<int> dst_dims = {0, 2, 3, 1};
          std::vector<index_t> output_shape =
              TransposeShape<index_t, index_t>(input->shape(), dst_dims);
          const float *input_ptr = input->data<float>();
          std::string nhwc_tensor_name =
              InternalTransformedName(input->name()) + "_nhwc";
          Tensor *nhwc_tensor = ws->CreateTensor(
              nhwc_tensor_name, GetCPUAllocator(), input->dtype());
          nhwc_tensor->Resize(output_shape);
          nhwc_tensor->set_data_format(DataFormat::NHWC);
          Tensor::MappingGuard nhwc_guard(nhwc_tensor);
          float *nhwc_ptr = nhwc_tensor->mutable_data<float>();
          RETURN_IF_ERROR(
              Transpose(input_ptr, input->shape(), dst_dims, nhwc_ptr));
          internal_tensor->Resize(output_shape);
          internal_tensor->set_data_format(DataFormat::NHWC);
          internal_tensor->CopyBytesWithMultiCore(nhwc_tensor->raw_data(),
                                                  nhwc_tensor->raw_size());
        } else if (model_type == ModelType::BERT) {
          // Transform filter
          internal_tensor->Resize(input->shape());
          const uint8_t *input_ptr = input->data<uint8_t>();
          Tensor::MappingGuard guard(internal_tensor);
          uint8_t *internal_ptr = internal_tensor->mutable_data<uint8_t>();
          memcpy(internal_ptr, input_ptr, input->raw_size());
        }
        // 2. convert the internal GPU Buffer to output.
        return kernel_->Compute(
            context, internal_tensor, type, wino_blk_size, output);
      }
    } else if (out_mem_type == MemoryType::CPU_BUFFER) {
      // 1. convert to the GPU Buffer with the output's data type.
      // Tensor internal_tensor(context->device()->allocator(),
      //                        dt,
      //                        false,
      //                        InternalTransformedName(input->name()));
      Tensor *internal_tensor =
          ws->CreateTensor(InternalTransformedName(input->name()),
                           context->device()->allocator(),
                           output->dtype());
      RETURN_IF_ERROR(kernel_->Compute(
          context, input, type, wino_blk_size, internal_tensor));
      // 2. convert the internal GPU Buffer to output.
      VLOG(INFO) << "Transform GPU Buffer " << internal_tensor->name()
                 << " to CPU Buffer " << output->name() << " with data type "
                 << dt << ", has_data_format: " << has_data_format
                 << ", shape: " << MakeString(internal_tensor->shape());
      if (has_data_format && internal_tensor->shape().size() == 4) {
        // NHWC -> NCHW
        std::vector<int> dst_dims = {0, 3, 1, 2};
        std::vector<index_t> output_shape = TransposeShape<index_t, index_t>(
            internal_tensor->shape(), dst_dims);
        std::string nhwc_tensor_name =
            InternalTransformedName(input->name()) + "_nhwc";
        Tensor *nhwc_tensor = ws->CreateTensor(
            nhwc_tensor_name, GetCPUAllocator(), output->dtype());
        nhwc_tensor->Resize(internal_tensor->shape());
        nhwc_tensor->set_data_format(DataFormat::NHWC);
        Tensor::MappingGuard guard(internal_tensor);
        nhwc_tensor->CopyBytesWithMultiCore(internal_tensor->raw_data(),
                                            internal_tensor->raw_size());
        const float *nhwc_ptr = nhwc_tensor->data<float>();
        output->Resize(output_shape);
        float *output_ptr = output->mutable_data<float>();
        return Transpose(nhwc_ptr, nhwc_tensor->shape(), dst_dims, output_ptr);
      } else {
        Tensor::MappingGuard guard(internal_tensor);
        const T *internal_ptr = internal_tensor->data<T>();
        output->Resize(internal_tensor->shape());
        T *output_ptr = output->mutable_data<T>();
        memcpy(output_ptr, internal_ptr, internal_tensor->raw_size());
        return VanState::SUCCEED;
      }
    } else {
      LOG(FATAL) << "Unexpected error: " << out_mem_type;
      return VanState::SUCCEED;
    }
  }

  VanState TransformSlice(OpContext *context,
                          const Tensor *input,
                          const OpenCLBufferType type,
                          const MemoryType out_mem_type,
                          const int wino_blk_size,
                          bool has_data_format,
                          Tensor *output) {
    NetworkController *ws = context->workspace();
    DataType dt = DataTypeToEnum<T>::value;
    MemoryType in_mem_type = input->memory_type();
    VLOG(INFO) << DEBUG_GPU << "Fucked Transform slice "
               << ", in memory type: " << in_mem_type
               << ", out memory type: " << out_mem_type
               << ", input name: " << input->name()
               << ", input shape: " << MakeString(input->shape())
               << ", output name: " << output->name()
               << ", has_data_format: " << has_data_format
               << ", opencl buffer: " << type;
    if (out_mem_type == MemoryType::GPU_IMAGE ||
        out_mem_type == MemoryType::GPU_BUFFER) {
      if (in_mem_type != MemoryType::CPU_BUFFER) {
        return kernel_->Compute(context, input, type, wino_blk_size, output);
      } else {
        // convert to the GPU Buffer with the input's data type.
        // 1. CPU buffer to GPU Buffer
        Tensor *internal_tensor =
            ws->CreateTensor(InternalTransformedName(input->name()),
                             context->device()->allocator(),
                             input->dtype());
        VLOG(INFO) << "Transform CPU Buffer " << input->name()
                   << " to GPU Buffer " << internal_tensor->name()
                   << " with data type " << dt;
        if (has_data_format && input->shape().size() == 4) {
          std::vector<index_t> output_shape = input->shape();
          internal_tensor->Resize(output_shape);
          internal_tensor->set_data_format(DataFormat::NHWC);
          Tensor::MappingGuard guard(input);
          internal_tensor->CopyBytesWithMultiCore(input->raw_data(),
                                                  input->raw_size());
        } else {
          LOG(FATAL) << "Unexpected error: " << out_mem_type;
        }
        // 2. convert the internal GPU Buffer to output.
        return kernel_->Compute(
            context, internal_tensor, type, wino_blk_size, output);
      }
    } else if (out_mem_type == MemoryType::CPU_BUFFER) {
      Tensor *internal_tensor =
          ws->CreateTensor(InternalTransformedName(input->name()),
                           context->device()->allocator(),
                           output->dtype());
      RETURN_IF_ERROR(kernel_->Compute(
          context, input, type, wino_blk_size, internal_tensor));
      VLOG(INFO) << "Transform GPU Buffer " << internal_tensor->name()
                 << " to CPU Buffer " << output->name() << " with data type "
                 << dt << ", has_data_format: " << has_data_format
                 << ", shape: " << MakeString(internal_tensor->shape());
      if (has_data_format && internal_tensor->shape().size() == 4) {
        output->Resize(internal_tensor->shape());
        output->set_data_format(DataFormat::NHWC);
        Tensor::MappingGuard guard(internal_tensor);
        output->CopyBytesWithMultiCore(internal_tensor->raw_data(),
                                       internal_tensor->raw_size());
        return VanState::SUCCEED;
      } else {
        LOG(FATAL) << "Unexpected error: " << out_mem_type;
        return VanState::UNSUPPORTED;
      }
    } else {
      LOG(FATAL) << "Unexpected error: " << out_mem_type;
      return VanState::SUCCEED;
    }
  }

private:
  std::string InternalTransformedName(const std::string &name) {
    const char *postfix = "_deepvan_identity_internal";
    return name + postfix;
  }
  const PruningType pruning_type_;

private:
  std::unique_ptr<OpenCLBufferTransformKernel> kernel_;
};

inline std::string TransformedFilterName(const std::string &name) {
  // TODO@vgod: This may create a conflict.
  const char *postfix = "_deepvan_identity_transformed";
  return name + postfix;
}

template <typename T>
VanState TransformFilter(deepvan::OpConstructContext *context,
                         OperatorProto *op_def,
                         const int input_idx,
                         const OpenCLBufferType buffer_type,
                         const MemoryType mem_type,
                         const int wino_blk_size = 0,
                         const PruningType pruning_type = PruningType::DENSE) {
  const DataType dt = DataTypeToEnum<T>::value;
  OpContext op_context(context->workspace(), context->device());
  NetworkController *ws = context->workspace();
  std::string input_name = op_def->input(input_idx);
  auto model_type = op_def->model_type();
  if (op_def->type() == "Conv2D") {
    model_type = ModelType::DEFAULT;
  }
  Tensor *input = ws->GetTensor(input_name);
  std::string output_name = TransformedFilterName(input_name);
  Tensor *output =
      ws->CreateTensor(output_name, context->device()->allocator(), dt, true);

  VLOG(3) << DEBUG_GPU << "Fucked TransformFilter"
          << ", in memory type: " << input->memory_type()
          << ", out memory type: " << mem_type << ", input_name: " << input_name
          << ", buffer type: " << buffer_type;
  // update the information
  op_def->set_input(input_idx, output_name);
  input->MarkUnused();
  return OpenCLBufferTransformer<T>(input->memory_type(),
                                    mem_type,
                                    pruning_type,
                                    model_type,
                                    buffer_type)
      .Transform(&op_context,
                 input,
                 buffer_type,
                 mem_type,
                 wino_blk_size,
                 DataFormat::DF_NONE,
                 model_type,
                 output);
}

} // namespace deepvan

#endif // DEEPVAN_BACKEND_OPENCL_BUFFER_TRANSFORMER_H_
