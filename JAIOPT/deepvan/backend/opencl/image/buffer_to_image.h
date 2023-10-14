#ifndef DEEPVAN_BACKEND_OPENCL_IMAGE_BUFFER_TO_IMAGE_H_
#define DEEPVAN_BACKEND_OPENCL_IMAGE_BUFFER_TO_IMAGE_H_

#include "deepvan/backend/opencl/buffer_transform_kernel.h"

#include <set>
#include <string>
#include <vector>

#include "deepvan/backend/opencl/helper.h"
#include "deepvan/backend/opencl/image/image_tensor_debug.h"
#include "deepvan/core/op_context.h"
#include "deepvan/core/tensor.h"
// #include "deepvan/backend/opencl/column/column_debug_util.h"
namespace deepvan {
namespace opencl {
namespace image {

template <typename T>
class BufferToImage : public OpenCLBufferTransformKernel {
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
VanState BufferToImage<T>::Compute(OpContext *context,
                                   const Tensor *input,
                                   const OpenCLBufferType type,
                                   const int wino_blk_size,
                                   Tensor *output) {
  auto formatted_buffer_shape = FormatBufferShape(input->shape(), type);
  std::vector<size_t> image_shape;
  OpenCLUtil::CalImage2DShape(formatted_buffer_shape,
                              type,
                              &image_shape,
                              wino_blk_size,
                              context->model_type());
  VLOG(3) << "Allocate OpenCL image: " << type;
  if (type == OpenCLBufferType::CONV2D_FILTER_BUFFER) {
    RETURN_IF_ERROR(output->Resize(input->shape()));
  } else {
    RETURN_IF_ERROR(output->ResizeImage(input->shape(), image_shape));
  }

  uint32_t gws[2] = {static_cast<uint32_t>(image_shape[0]),
                     static_cast<uint32_t>(image_shape[1])};
  if (type == MATMUL_FILTER) {
    gws[0] = 1;              // w
    gws[1] = image_shape[1]; // h
  }
  std::string kernel_name;
  switch (type) {
  case CONV2D_FILTER: kernel_name = "filter_buffer_to_image"; break;
  case DW_CONV2D_FILTER: kernel_name = "dw_filter_buffer_to_image"; break;
  case BUFFER_2_BUFFER: kernel_name = "in_out_buffer_to_image"; break;
  case IN_OUT_CHANNEL: kernel_name = "in_out_buffer_to_image"; break;
  case ARGUMENT: kernel_name = "arg_buffer_to_image"; break;
  case IN_OUT_HEIGHT: kernel_name = "in_out_height_buffer_to_image"; break;
  case IN_OUT_WIDTH: kernel_name = "in_out_width_buffer_to_image"; break;
  case WEIGHT_HEIGHT: kernel_name = "weight_height_buffer_to_image"; break;
  case WEIGHT_WIDTH: kernel_name = "weight_width_buffer_to_image"; break;
  case MATMUL_FILTER: kernel_name = "matmul_filter_buffer_to_image"; break;
  case CONV2D_FILTER_BUFFER: kernel_name = "filter_buffer_to_buffer"; break;
  case WINOGRAD_FILTER: {
    std::stringstream ss_tmp;
    gws[1] /= (wino_blk_size + 2) * (wino_blk_size + 2);
    ss_tmp << "winograd_filter_buffer_to_image_" << wino_blk_size << "x"
           << wino_blk_size;
    kernel_name = ss_tmp.str();
    break;
  }
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
    RETURN_IF_ERROR(runtime->BuildKernel(
        "buffer_to_image", obfuscated_kernel_name, built_options, &kernel_));
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
    if (type == CONV2D_FILTER || type == CONV2D_FILTER_BUFFER) {
      const index_t inner_size = input->dim(1) * input->dim(2) * input->dim(3);
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(0))); // 32
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(2))); // 3
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(3))); // 3
      kernel_.setArg(idx++, static_cast<uint32_t>(inner_size));    // 16*3*3
    } else if (type == DW_CONV2D_FILTER || type == WEIGHT_HEIGHT) {
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(0)));
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(1)));
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(2)));
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(3)));
    } else if (type == ARGUMENT) {
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(0)));
    } else if (type == MATMUL_FILTER) {
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(0))); // row 4096
      kernel_.setArg(idx++, static_cast<uint32_t>(input->dim(1))); // col 25088
    } else {
      kernel_.setArg(idx++, static_cast<uint32_t>(formatted_buffer_shape[1]));
      kernel_.setArg(idx++, static_cast<uint32_t>(formatted_buffer_shape[2]));
      kernel_.setArg(idx++, static_cast<uint32_t>(formatted_buffer_shape[3]));
    }
    if (type == CONV2D_FILTER_BUFFER) {
      kernel_.setArg(idx++, *(output->opencl_buffer()));
    } else {
      kernel_.setArg(idx++, *(output->opencl_image()));
    }
    input_shape_ = input->shape();
  }

  const uint32_t kwg_size =
      static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  uint32_t lws0 = type == MATMUL_FILTER ? 1 : 16;
  uint32_t lws1 = type == MATMUL_FILTER ? 64 : kwg_size / 16;
  const std::vector<uint32_t> lws = {lws0, lws1};
  RETURN_IF_ERROR(Run2DKernel(runtime, kernel_, gws, lws, context->future()));
  OUT_OF_RANGE_VALIDATION;
  // std::vector<size_t> output_pitch = {0, 0};
  // std::vector<size_t> output_pitch0 = {0, 0};
  // LOG(INFO) << output->dim_size() << "  " << MakeString(output->shape()) <<
  // output->name() << " " << MakeString(input->shape()) << " " <<kernel_name;

  // if (output->name() == "const_fold_opt__497_deepvan_identity_transformed"){
  //   LOG(INFO) << output->dim_size() << "  " << MakeString(output->shape()) <<
  //   output->name() << " " << MakeString(input->shape()) << " " <<kernel_name;
  //   LOG(INFO) << gws[0] << " " << gws[1];
  //       // image::Write4DTensorToFile<float>(output, output_pitch, "filter",
  //       true);
  //   // image::Write4DTensorToFile<float>(input, output_pitch0, "filter_0",
  //   true);

  //   // image::WriteTensor2File(output, output_pitch, "filter", true);

  //   Tensor::MappingGuard output_guard(output);
  //   // column::Show2DResult(output, " ", false, 32/4,16*9);
  //   std::stringstream ss;
  //   auto output_data = output->data<float>();
  //   // for(int i=0; i<4*4*9*;i++){
  //   for(int i=0; i<16*32*9;i++){
  //     ss << output_data[i] << ", ";
  //   }
  //   LOG(INFO) << ss.str();

  //   // column::Show4DResult(output, " ", 0,0);
  // }

  return VanState::SUCCEED;
}

} // namespace image
} // namespace opencl
} // namespace deepvan

#endif // DEEPVAN_BACKEND_OPENCL_IMAGE_BUFFER_TO_IMAGE_H_
