#include "deepvan/backend/opencl/kernel_util.h"

namespace deepvan {
namespace opencl {

namespace kernel_util {
std::vector<uint32_t> LocalWS(OpenCLRuntime *runtime,
                              const uint32_t *gws,
                              const uint32_t kwg_size) {
  std::vector<uint32_t> lws(4, 0);
  if (kwg_size == 0) {
    lws[0] = lws[1] = lws[2] = 1;
  } else {
    uint64_t cache_size = runtime->device_global_mem_cache_size();
    uint32_t base = std::max<uint32_t>(cache_size / kBaseGPUMemCacheSize, 1);
    lws[1] = std::min<uint32_t>(gws[1], kwg_size);
    lws[2] = min2<uint32_t>(gws[2], base, kwg_size / lws[1]);
    const uint32_t lws_size = lws[1] * lws[2];
    lws[0] = gws[0] / 4;
    if (lws[0] == 0) {
      lws[0] = gws[0];
    }
    lws[0] = clamp<uint32_t>(lws[0], 1, kwg_size / lws_size);
  }
  return lws;
}

} // namespace kernel_util

VanState ChannelReorder(OpContext *context,
                        cl::Kernel &kernel_,
                        bool &init_pad_kernel,
                        const Tensor *src,
                        const Tensor *filter,
                        std::vector<int> &paddings,
                        Tensor *dst,
                        StatsFuture *future) {
  auto first = NowMicros();
  auto src_shape = src->shape();
  auto dt = src->dtype();
  const size_t in_channel = src_shape[1];
  const size_t in_height = src_shape[2];
  const size_t in_width = src_shape[3];

  const size_t out_channel_width = RoundUpDiv4(in_width) + 1;
  std::vector<size_t> out_image_shape(2);
  PatternUtil::CalImagePadShape(
      src_shape, {paddings[0], paddings[1]}, out_image_shape);
  uint32_t gws[3] = {static_cast<uint32_t>(in_channel),
                     static_cast<uint32_t>(out_image_shape[0] / in_channel),
                     static_cast<uint32_t>(out_image_shape[1])};
  dst->ResizeImage(src_shape, out_image_shape);

  std::string kernel_name;
  if (paddings[0] == 2 && paddings[1] == 2) {
    kernel_name = "reorder_nchw_3x3";
  } else {
    CONDITIONS(false, "DeepVan don't support other padding except {[3, 3]}");
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
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(filter->dtype()));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(filter->dtype()));
    RETURN_IF_ERROR(runtime->BuildKernel(
        "pattern_pad", obfuscated_kernel_name, built_options, &kernel_));
  }
  OUT_OF_RANGE_INIT(kernel_);

  if (!init_pad_kernel) {
    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARGS(kernel_);
    SET_2D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(src->opencl_image()));

    kernel_.setArg(idx++, *(filter->opencl_buffer()));
    kernel_.setArg(idx++, static_cast<uint32_t>(in_height));
    kernel_.setArg(idx++, static_cast<uint32_t>(in_width));
    kernel_.setArg(idx++, static_cast<uint32_t>(out_image_shape[1]));
    kernel_.setArg(idx++, static_cast<uint32_t>(out_channel_width));
    kernel_.setArg(idx++, *(dst->opencl_image()));
    init_pad_kernel = true;
  }

  uint32_t kwg_size_ =
      static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  const std::vector<uint32_t> lws =
      kernel_util::LocalWS(runtime, gws, kwg_size_);

  cl::Event event;
  cl_int error;
  first = NowMicros();
  error = runtime->command_queue().enqueueNDRangeKernel(
      kernel_,
      cl::NullRange,
      cl::NDRange(gws[0], gws[1], gws[2]),
      cl::NDRange(lws[0], lws[1], lws[2]),
      nullptr,
      &event);
  VLOG(1) << DEBUG_GPU << "Enqueue pad kernel consumes: " << NowMicros() - first
          << " us";
  CL_RET_STATUS(error);
  OUT_OF_RANGE_VALIDATION;
  if (future != nullptr) {
    future->wait_fn = [runtime, event](CallStats *stats) {
      event.wait();
      if (stats != nullptr) {
        runtime->GetCallStats(event, stats);
      }
    };
  }
  return VanState::SUCCEED;
}

} // namespace opencl

} // namespace deepvan
