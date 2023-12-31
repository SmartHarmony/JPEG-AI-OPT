#include "deepvan/backend/common/activation_type.h"
#include "deepvan/backend/common/conv_pool_2d_util.h"
#include "deepvan/backend/opencl/helper.h"
#include "deepvan/core/op_context.h"
#include "deepvan/core/runtime/opencl/opencl_runtime.h"
#include "deepvan/utils/math.h"
#include "deepvan/utils/memory.h"

namespace deepvan {
namespace opencl {
namespace image {

namespace {
VanState WinogradInputTransform(OpContext *context,
                                cl::Kernel *kernel,
                                const Tensor *input_tensor,
                                const DataType dt,
                                const int *paddings,
                                const index_t round_h,
                                const index_t round_w,
                                const int wino_blk_size,
                                const bool input_changed,
                                Tensor *output_tensor,
                                uint32_t *kwg_size,
                                StatsFuture *future) {
  OpenCLRuntime *runtime = context->device()->gpu_runtime()->opencl_runtime();
  const index_t out_width = output_tensor->dim(2);

  OUT_OF_RANGE_DEFINITION;

  if (kernel->get() == nullptr) {
    std::string obfuscated_kernel_name;
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG;
    NON_UNIFORM_WG_CONFIG;
    if (wino_blk_size == 4) {
      obfuscated_kernel_name =
          DEEPVAN_OBFUSCATE_SYMBOL("winograd_transform_4x4");
      built_options.emplace("-Dwinograd_transform_4x4=" +
                            obfuscated_kernel_name);
    } else if (wino_blk_size == 2) {
      obfuscated_kernel_name =
          DEEPVAN_OBFUSCATE_SYMBOL("winograd_transform_2x2");
      built_options.emplace("-Dwinograd_transform_2x2=" +
                            obfuscated_kernel_name);
    } else {
      CONDITIONS(false, "deepvan only supports 4x4 and 2x2 gpu winograd.");
      return VanState::SUCCEED;
    }
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
    RETURN_IF_ERROR(runtime->BuildKernel(
        "winograd_transform", obfuscated_kernel_name, built_options, kernel));

    *kwg_size =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(*kernel));
  }

  const uint32_t gws[2] = {
      static_cast<uint32_t>(out_width),
      static_cast<uint32_t>(RoundUpDiv4(input_tensor->dim(3)))};
  OUT_OF_RANGE_INIT(*kernel);
  if (input_changed) {
    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARGS(*kernel);
    SET_2D_GWS_ARGS(*kernel, gws);
    kernel->setArg(idx++, *(input_tensor->opencl_image()));
    kernel->setArg(idx++, *(output_tensor->opencl_image()));
    kernel->setArg(idx++, static_cast<uint32_t>(input_tensor->dim(1)));
    kernel->setArg(idx++, static_cast<uint32_t>(input_tensor->dim(2)));
    kernel->setArg(idx++, static_cast<uint32_t>(input_tensor->dim(3)));
    kernel->setArg(idx++, static_cast<uint32_t>(round_h * round_w));
    kernel->setArg(idx++, static_cast<uint32_t>(round_w));
    kernel->setArg(idx++, static_cast<uint32_t>(paddings[0] / 2));
    kernel->setArg(idx++, static_cast<uint32_t>(paddings[1] / 2));
  }

  const std::vector<uint32_t> lws = {*kwg_size / 8, 8, 0};
  RETURN_IF_ERROR(Run2DKernel(runtime, *kernel, gws, lws, future));

  OUT_OF_RANGE_VALIDATION;
  return VanState::SUCCEED;
}

VanState WinogradOutputTransform(OpContext *context,
                                 cl::Kernel *kernel,
                                 const Tensor *input_tensor,
                                 const Tensor *bias,
                                 const DataType dt,
                                 const index_t round_h,
                                 const index_t round_w,
                                 const int wino_blk_size,
                                 const ActivationType activation,
                                 const float relux_max_limit,
                                 const float leakyrelu_coefficient,
                                 const bool input_changed,
                                 Tensor *output_tensor,
                                 uint32_t *kwg_size,
                                 StatsFuture *future) {
  OpenCLRuntime *runtime = context->device()->gpu_runtime()->opencl_runtime();
  auto &output_shape = output_tensor->shape();

  OUT_OF_RANGE_DEFINITION;
  if (kernel->get() == nullptr) {
    std::string obfuscated_kernel_name;
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG;
    NON_UNIFORM_WG_CONFIG;
    if (wino_blk_size == 4) {
      obfuscated_kernel_name =
          DEEPVAN_OBFUSCATE_SYMBOL("winograd_inverse_transform_4x4");
      built_options.emplace("-Dwinograd_inverse_transform_4x4=" +
                            obfuscated_kernel_name);
    } else if (wino_blk_size == 2) {
      obfuscated_kernel_name =
          DEEPVAN_OBFUSCATE_SYMBOL("winograd_inverse_transform_2x2");
      built_options.emplace("-Dwinograd_inverse_transform_2x2=" +
                            obfuscated_kernel_name);
    } else {
      CONDITIONS(false, "deepvan only supports 4x4 and 2x2 gpu winograd.");
      return VanState::SUCCEED;
    }

    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
    built_options.emplace(bias != nullptr ? "-DBIAS" : "");
    switch (activation) {
    case NOOP: break;
    case RELU: built_options.emplace("-DUSE_RELU"); break;
    case RELUX: built_options.emplace("-DUSE_RELUX"); break;
    case PRELU: built_options.emplace("-DUSE_PRELU"); break;
    case TANH: built_options.emplace("-DUSE_TANH"); break;
    case SIGMOID: built_options.emplace("-DUSE_SIGMOID"); break;
    case LEAKYRELU: built_options.emplace("-DUSE_LEAKYRELU"); break;
    case SWISH: built_options.emplace("-DUSE_SWISH"); break;
    default: LOG(FATAL) << "Unknown activation type: " << activation;
    }

    RETURN_IF_ERROR(runtime->BuildKernel(
        "winograd_transform", obfuscated_kernel_name, built_options, kernel));

    *kwg_size =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(*kernel));
  }

  const uint32_t gws[2] = {
      static_cast<uint32_t>(input_tensor->dim(2)),
      static_cast<uint32_t>(RoundUpDiv4(input_tensor->dim(1)))};
  OUT_OF_RANGE_INIT(*kernel);
  if (input_changed) {
    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARGS(*kernel);
    SET_2D_GWS_ARGS(*kernel, gws);
    kernel->setArg(
        idx++,
        *(static_cast<const cl::Image2D *>(input_tensor->opencl_image())));
    if (bias != nullptr) {
      kernel->setArg(idx++,
                     *(static_cast<const cl::Image2D *>(bias->opencl_image())));
    }
    kernel->setArg(
        idx++, *(static_cast<cl::Image2D *>(output_tensor->opencl_image())));
    kernel->setArg(idx++, static_cast<uint32_t>(output_shape[1]));
    kernel->setArg(idx++, static_cast<uint32_t>(output_shape[2]));
    kernel->setArg(idx++, static_cast<uint32_t>(round_h * round_w));
    kernel->setArg(idx++, static_cast<uint32_t>(round_w));
    kernel->setArg(idx++, relux_max_limit);
    kernel->setArg(idx++, leakyrelu_coefficient);
  }
  const std::vector<uint32_t> lws = {*kwg_size / 8, 8, 0};
  RETURN_IF_ERROR(Run2DKernel(runtime, *kernel, gws, lws, future));

  OUT_OF_RANGE_VALIDATION;
  return VanState::SUCCEED;
}
} // namespace

extern VanState WinogradConv2dK3x3S1(OpContext *context,
                                     cl::Kernel *kernels[3],
                                     const Tensor *input,
                                     const Tensor *filter,
                                     const Tensor *bias,
                                     const int *paddings,
                                     const ActivationType activation,
                                     const float relux_max_limit,
                                     const float leakyrelu_coefficient,
                                     const DataType dt,
                                     const int wino_blk_size,
                                     std::vector<index_t> *prev_input_shape,
                                     Tensor *output,
                                     uint32_t *kwg_size[3],
                                     const std::string &op_name) {
  VLOG(INFO) << "Running Conv2d: " << op_name;
  OpenCLRuntime *runtime = context->device()->gpu_runtime()->opencl_runtime();
  ScratchImageManager *scratch_manager =
      context->device()->gpu_runtime()->scratch_image_manager();
  StatsFuture t_input_future, mm_future, t_output_future;
  bool input_changed = !IsVecEqual(*prev_input_shape, input->shape());
  *prev_input_shape = input->shape();

  auto output_shape = output->shape();
  const index_t round_h = (output_shape[1] + wino_blk_size - 1) / wino_blk_size;
  const index_t round_w = (output_shape[2] + wino_blk_size - 1) / wino_blk_size;
  const index_t out_width = input->dim(0) * round_h * round_w;

  const index_t blk_sqr = (wino_blk_size + 2) * (wino_blk_size + 2);

  index_t in_channel = input->dim(3);
  index_t out_channel = output->dim(3);

  // 0. transform input
  // input(NHWC) -> t_input(blk_sqr, in_channel, out_width)
  std::vector<index_t> t_input_shape = {blk_sqr, in_channel, out_width};
  std::vector<index_t> padded_t_input_shape = {
      t_input_shape[0], t_input_shape[1], t_input_shape[2], 1};
  std::vector<size_t> t_input_image_shape;
  OpenCLUtil::CalImage2DShape(padded_t_input_shape,
                              OpenCLBufferType::IN_OUT_HEIGHT,
                              &t_input_image_shape, 0, context->model_type());
  ScratchImage transformed_input_image(scratch_manager);
  std::unique_ptr<Tensor> transformed_input = make_unique<Tensor>(
      transformed_input_image.Scratch(
          context->device()->allocator(), t_input_image_shape, dt),
      dt);
  RETURN_IF_ERROR(
      transformed_input->ResizeImage(t_input_shape, t_input_image_shape));
  RETURN_IF_ERROR(WinogradInputTransform(context,
                                         kernels[0],
                                         input,
                                         dt,
                                         paddings,
                                         round_h,
                                         round_w,
                                         wino_blk_size,
                                         input_changed,
                                         transformed_input.get(),
                                         kwg_size[0],
                                         &t_input_future));

  // 1. mat mul
  // t_filter(blk_sqr, out_chan, in_chan)*t_input(blk_sqr, in_chan, out_width)
  //     -> t_output (blk_sqr, out_chan, out_width)
  std::vector<index_t> mm_output_shape = {blk_sqr, out_channel, out_width};

  std::vector<index_t> padded_mm_output_shape = {
      mm_output_shape[0], mm_output_shape[1], mm_output_shape[2], 1};
  std::vector<size_t> mm_output_image_shape;
  OpenCLUtil::CalImage2DShape(padded_mm_output_shape,
                              OpenCLBufferType::IN_OUT_HEIGHT,
                              &mm_output_image_shape, 0, context->model_type());

  ScratchImage mm_output_image(scratch_manager);
  std::unique_ptr<Tensor> mm_output = make_unique<Tensor>(
      mm_output_image.Scratch(
          context->device()->allocator(), mm_output_image_shape, dt),
      dt);
  RETURN_IF_ERROR(
      mm_output->ResizeImage(mm_output_shape, mm_output_image_shape));

  const index_t height_blocks = RoundUpDiv4(mm_output_shape[1]);
  const index_t width_blocks = RoundUpDiv4(mm_output_shape[2]);
  const uint32_t gws[2] = {
      static_cast<uint32_t>(width_blocks),
      static_cast<uint32_t>(height_blocks * blk_sqr),
  };

  OUT_OF_RANGE_DEFINITION;

  if (kernels[1]->get() == nullptr) {
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG;
    NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = DEEPVAN_OBFUSCATE_SYMBOL("matmul");
    built_options.emplace("-Dmatmul=" + kernel_name);
    // built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    // built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
    RETURN_IF_ERROR(
        runtime->BuildKernel("matmul", kernel_name, built_options, kernels[1]));

    *kwg_size[1] =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(*kernels[1]));
  }
  OUT_OF_RANGE_INIT(*kernels[1]);
  uint32_t idx = 0;
  OUT_OF_RANGE_SET_ARGS(*kernels[1]);
  SET_2D_GWS_ARGS(*kernels[1], gws);
  kernels[1]->setArg(idx++, *(filter->opencl_image()));
  kernels[1]->setArg(idx++, *(transformed_input->opencl_image()));
  kernels[1]->setArg(idx++, *(mm_output->opencl_image()));
  kernels[1]->setArg(idx++, static_cast<int>(mm_output_shape[1]));
  kernels[1]->setArg(idx++, static_cast<int>(mm_output_shape[2]));
  kernels[1]->setArg(idx++, static_cast<int>(in_channel));
  kernels[1]->setArg(idx++, static_cast<int>(height_blocks));
  kernels[1]->setArg(idx++, static_cast<int>(RoundUpDiv4(in_channel)));
  const std::vector<uint32_t> lws = {*kwg_size[1] / 64, 64, 0};
  RETURN_IF_ERROR(Run2DKernel(runtime, *kernels[1], gws, lws, &mm_future));

  OUT_OF_RANGE_VALIDATION;

  // 2. transform output
  // t_output (blk_sqr, out_chan, out_width) -> output(NHWC)
  RETURN_IF_ERROR(WinogradOutputTransform(context,
                                          kernels[2],
                                          mm_output.get(),
                                          bias,
                                          dt,
                                          round_h,
                                          round_w,
                                          wino_blk_size,
                                          activation,
                                          relux_max_limit,
                                          leakyrelu_coefficient,
                                          input_changed,
                                          output,
                                          kwg_size[2],
                                          &t_output_future))
  MergeMultipleFutureWaitFn({t_input_future, mm_future, t_output_future},
                            context->future());
  return VanState::SUCCEED;
}

} // namespace image
} // namespace opencl
} // namespace deepvan
