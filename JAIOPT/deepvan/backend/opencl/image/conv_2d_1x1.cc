#include "deepvan/backend/common/activation_type.h"
#include "deepvan/backend/opencl/helper.h"
#include "deepvan/core/op_context.h"
#include "deepvan/core/runtime/opencl/opencl_runtime.h"
#include "deepvan/core/types.h"
#include "deepvan/utils/math.h"

namespace deepvan {
namespace opencl {
namespace image {

namespace {
// (inputs + weights + outputs) * array_size * sizeof(float)
const uint32_t kernel_cache_size = (4 + 4 + 4) * 4 * 4;
const uint32_t lws_limit = 128;
std::vector<uint32_t> LocalWS(OpenCLRuntime *runtime,
                              const uint32_t *gws,
                              const uint32_t kwg_size) {
  std::vector<uint32_t> lws(4, 0);
  if (kwg_size == 0) {
    lws[0] = lws[1] = lws[2] = 1;
  } else {
    uint64_t cache_size = runtime->device_global_mem_cache_size();
    uint32_t compute_units = runtime->device_compute_units();
    const uint32_t base =
        std::max<uint32_t>(cache_size / kBaseGPUMemCacheSize, 1);
    lws[1] = std::min<uint32_t>(gws[1], kwg_size);
    if (lws[1] >= base) {
      lws[0] = std::min<uint32_t>(gws[0], base);
    } else if ((1 < lws[1] && lws[1] < base) && gws[0] >= lws_limit) {
      lws[0] = std::min<uint32_t>(gws[0], base);
    } else {
      lws[0] = gws[0] / 8;
      if (lws[0] < base) {
        lws[0] = std::max<uint32_t>(gws[0] / 4, base);
      }
    }
    lws[0] = std::min<uint32_t>(lws[0], kwg_size / lws[1]);
    const uint32_t lws_size = lws[0] * lws[1];
    lws[2] = std::min<uint32_t>(
        (cache_size / kernel_cache_size / lws_size / compute_units) * 8,
        gws[2]);
    if (lws[2] == 0) {
      lws[2] = std::min<uint32_t>(gws[2], base);
    }
    lws[2] =
        std::max<uint32_t>(std::min<uint32_t>(lws[2], kwg_size / lws_size), 1);
  }
  return lws;
}

} // namespace

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
                           const Tensor *elt) {
  UNUSED_VARIABLE(padding);
  UNUSED_VARIABLE(dilations);
  VLOG(INFO) << "Running Conv2d: " << op_name;
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);
  const index_t input_batch = input->dim(0);
  const index_t input_height = input->dim(1);
  const index_t input_width = input->dim(2);
  const index_t input_channels = input->dim(3);

  auto CheckDebugLayer = [&]() -> bool { return true; };

  auto GetKernelConfiguration = [&]() -> KernelConfiguration {
    std::string kernel_name = "conv_2d_1x1";
    std::string program_name = "conv_2d_1x1";
    index_t channel_blocks = RoundUpDiv4(channels);
    index_t width_blocks = RoundUpDiv4(width);
    index_t height_blocks = height * batch;
    return KernelConfiguration(
        kernel_name, program_name, channel_blocks, width_blocks, height_blocks);
  };

  KernelConfiguration kernel_conf = GetKernelConfiguration();
  const index_t input_channel_blocks = RoundUpDiv4(input_channels);

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  OUT_OF_RANGE_DEFINITION;

  if (kernel->get() == nullptr) {
    CONDITIONS(input_batch == batch);
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG;
    NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = DEEPVAN_OBFUSCATE_SYMBOL(kernel_conf.kernel_name);
    built_options.emplace("-D" + kernel_conf.kernel_name + "=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" +
                          DtToCLDt(dt)); // DtToUpCompatibleCLDt(dt)
    built_options.emplace("-DCMD_DATA_TYPE=" +
                          DtToCLCMDDt(dt)); // DtToUpCompatibleCLCMDDt(dt)
    if (bias != nullptr) {
      built_options.emplace("-DBIAS");
    }
    built_options.emplace(elt != nullptr ? "-DELEMENT" : "");
    switch (activation) {
    case NOOP: break;
    case RELU: built_options.emplace("-DUSE_RELU"); break;
    case RELUX: built_options.emplace("-DUSE_RELUX"); break;
    case TANH: built_options.emplace("-DUSE_TANH"); break;
    case SIGMOID: built_options.emplace("-DUSE_SIGMOID"); break;
    case LEAKYRELU: built_options.emplace("-DUSE_LEAKYRELU"); break;
    case SWISH: built_options.emplace("-DUSE_SWISH"); break;
    default: LOG(FATAL) << "Unknown activation type: " << activation;
    }

    RETURN_IF_ERROR(runtime->BuildKernel(
        kernel_conf.program_name, kernel_name, built_options, kernel));

    *kwg_size =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(*kernel));
  }

  const uint32_t gws[3] = {static_cast<uint32_t>(kernel_conf.channel_blocks),
                           static_cast<uint32_t>(kernel_conf.width_blocks),
                           static_cast<uint32_t>(kernel_conf.height_blocks)};
  OUT_OF_RANGE_INIT(*kernel);

  // Support different input size
  if (!IsVecEqual(*prev_input_shape, input->shape())) {
    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARGS(*kernel);
    SET_3D_GWS_ARGS(*kernel, gws);
    kernel->setArg(idx++, *(input->opencl_image()));
    kernel->setArg(idx++, *(filter->opencl_image()));
    if (bias != nullptr) {
      kernel->setArg(idx++, *(bias->opencl_image()));
    }
    if (elt != nullptr) {
      kernel->setArg(idx++, *(elt->opencl_image()));
    }
    kernel->setArg(idx++, *(output->opencl_image()));
    // FIXME handle flexable data type: half not supported
    kernel->setArg(idx++, relux_max_limit);
    kernel->setArg(idx++, leakyrelu_coefficient);
    kernel->setArg(idx++, static_cast<int>(input_height));
    kernel->setArg(idx++, static_cast<int>(input_width));
    kernel->setArg(idx++, static_cast<int>(input_channel_blocks));
    kernel->setArg(idx++, static_cast<int>(height));
    kernel->setArg(idx++, static_cast<int>(width));
    kernel->setArg(idx++, stride);

    *prev_input_shape = input->shape();
  }

  // std::vector<uint32_t> lws = LocalWS(runtime, gws, *kwg_size);
  std::vector<uint32_t> lws = {4,4,16};
  // LOG(INFO) << gws[0] << " " << gws[1] << " " << gws[2] << " " << lws[0] << " " << lws[1] << " " << lws[2];
  RETURN_IF_ERROR(Run3DKernel(runtime, *kernel, gws, lws, context->future()));
  OUT_OF_RANGE_VALIDATION;
  return VanState::SUCCEED;
}

} // namespace image
} // namespace opencl
} // namespace deepvan
