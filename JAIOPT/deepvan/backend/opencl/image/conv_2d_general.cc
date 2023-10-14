#include "deepvan/backend/common/activation_type.h"
#include "deepvan/backend/opencl/helper.h"
#include "deepvan/core/op_context.h"
#include "deepvan/core/runtime/opencl/opencl_runtime.h"
#include "deepvan/utils/math.h"

namespace deepvan {
namespace opencl {
namespace image {

namespace {
// (inputs + weights + outputs) * array_size * sizeof(float)
const uint32_t kernel_cache_size = (4 + 4 + 4) * 4 * 4;
const uint32_t lws_limit = 20;
std::vector<uint32_t> LocalWS(OpenCLRuntime *runtime,
                              const uint32_t *gws,
                              const uint32_t kernel_size,
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
    lws[0] = gws[0] / 4;
    if (lws[0] == 0) {
      lws[0] = gws[0];
    }
    lws[0] = std::min<uint32_t>(lws[0], kwg_size / lws[1]);
    const uint32_t lws_size = lws[0] * lws[1];
    lws[2] = std::min<uint32_t>((cache_size / kernel_cache_size / kernel_size /
                                 lws_size / compute_units) *
                                    8,
                                gws[2]);
    if (lws[2] == 0) {
      if (gws[2] < lws_limit) {
        lws[2] = gws[2];
      } else {
        lws[2] = base;
      }
    }
    lws[2] =
        std::max<uint32_t>(std::min<uint32_t>(lws[2], kwg_size / lws_size), 1);
  }
  return lws;
}

} // namespace

extern VanState Conv2d(OpContext *context,
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
  VLOG(INFO) << "Running Conv2d: " << op_name;
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);
  const index_t input_channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t input_channel_blocks = RoundUpDiv4(input_channels);
  const index_t width_blocks = RoundUpDiv4(width);

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  OUT_OF_RANGE_DEFINITION;

  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG;
    NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = DEEPVAN_OBFUSCATE_SYMBOL("conv_2d");
    built_options.emplace("-Dconv_2d=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
    built_options.emplace(bias != nullptr ? "-DBIAS" : "");
    built_options.emplace(elt != nullptr ? "-DELEMENT" : "");
    switch (activation) {
    case NOOP: break;
    case RELU: built_options.emplace("-DUSE_RELU"); break;
    case RELUX: built_options.emplace("-DUSE_RELUX"); break;
    case TANH: built_options.emplace("-DUSE_TANH"); break;
    case SIGMOID: built_options.emplace("-DUSE_SIGMOID"); break;
    case LEAKYRELU: built_options.emplace("-DUSE_LEAKYRELU"); break;
    default: LOG(FATAL) << "Unknown activation type: " << activation;
    }

    RETURN_IF_ERROR(
        runtime->BuildKernel("conv_2d", kernel_name, built_options, kernel));

    *kwg_size =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(*kernel));
  }

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width_blocks),
                           static_cast<uint32_t>(height * batch)};
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
    kernel->setArg(idx++, relux_max_limit);
    kernel->setArg(idx++, leakyrelu_coefficient);
    kernel->setArg(idx++, static_cast<uint32_t>(input->dim(1)));
    kernel->setArg(idx++, static_cast<uint32_t>(input->dim(2)));
    kernel->setArg(idx++, static_cast<uint32_t>(input_channel_blocks));
    kernel->setArg(idx++, static_cast<uint32_t>(height));
    kernel->setArg(idx++, static_cast<uint32_t>(width));
    kernel->setArg(idx++, static_cast<uint32_t>(filter->dim(2)));
    kernel->setArg(idx++, static_cast<uint32_t>(filter->dim(3)));
    kernel->setArg(idx++, static_cast<uint32_t>(stride));
    kernel->setArg(idx++, padding[0] / 2);
    kernel->setArg(idx++, padding[1] / 2);
    kernel->setArg(idx++, dilations[0]);
    kernel->setArg(idx++, dilations[1]);

    *prev_input_shape = input->shape();
  }

  std::vector<uint32_t> lws =
      LocalWS(runtime, gws, filter->dim(2) * filter->dim(3), *kwg_size);
  RETURN_IF_ERROR(Run3DKernel(runtime, *kernel, gws, lws, context->future()));

  OUT_OF_RANGE_VALIDATION;
  return VanState::SUCCEED;
}

} // namespace image
} // namespace opencl
} // namespace deepvan
