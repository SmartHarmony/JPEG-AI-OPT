#ifndef DEEPVAN_BACKEND_OPENCL_IMAGE_GEMV_H_
#define DEEPVAN_BACKEND_OPENCL_IMAGE_GEMV_H_

#include "deepvan/backend/opencl/gemv.h"

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "deepvan/backend/opencl/helper.h"
#include "deepvan/core/op_context.h"
#include "deepvan/core/tensor.h"

namespace deepvan {
namespace opencl {
namespace image {

template <typename T>
class GEMVKernel : public OpenCLGEMVKernel {
public:
  explicit GEMVKernel(const ActivationType activation,
                      const float relux_max_limit,
                      const float leakyrelu_coefficient)
      : OpenCLGEMVKernel(activation, relux_max_limit, leakyrelu_coefficient) {}

  VanState Compute(OpContext *context,
                   const Tensor *A,
                   const Tensor *B,
                   const Tensor *bias,
                   Tensor *C,
                   bool transpose_a,
                   bool transpose_b) override;

private:
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  bool has_kernel_init_ = false;
};

template <typename T>
VanState GEMVKernel<T>::Compute(OpContext *context,
                                const Tensor *A, // input
                                const Tensor *B, // weight
                                const Tensor *bias,
                                Tensor *C, // result
                                bool transpose_a,
                                bool transpose_b) {
  UNUSED_VARIABLE(bias);
  const index_t lhs_rank = A->dim_size();
  const index_t lhs_rows = A->dim(lhs_rank - 2);
  const index_t lhs_cols = A->dim(lhs_rank - 1);
  const index_t rhs_rank = B->dim_size();
  const index_t rhs_rows = B->dim(rhs_rank - 2);
  const index_t rhs_cols = B->dim(rhs_rank - 1);
  const bool has_opencl_image = B->has_opencl_image();

  const index_t rows = transpose_a ? lhs_cols : lhs_rows;
  const index_t cols = transpose_b ? rhs_rows : rhs_cols;
  const index_t depth = transpose_a ? lhs_rows : lhs_cols;

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  std::vector<index_t> output_shape = {rows, cols};
  std::vector<size_t> output_image_shape = {static_cast<size_t>(cols / 4),
                                            static_cast<size_t>(rows)};
  RETURN_IF_ERROR(C->ResizeImage(output_shape, output_image_shape));
  const DataType dt = A->dtype();

  const uint32_t gws = static_cast<uint32_t>(output_image_shape[0] >> 1);
  std::uint32_t lws = 16; // w, h

  VLOG(INFO) << DEBUG_GPU << "GEMV output shape: " << MakeString(output_shape)
             << ", output image shape: " << MakeString(output_image_shape)
             << ", gws: " << gws << ", has_opencl_image: " << has_opencl_image;

  OUT_OF_RANGE_DEFINITION;
  if (!has_kernel_init_) {
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG;
    NON_UNIFORM_WG_CONFIG;
    std::string kernel_name =
        DEEPVAN_OBFUSCATE_SYMBOL(has_opencl_image ? "gemv" : "gemv_hybrid");
    built_options.emplace((has_opencl_image ? "-Dgemv=" : "-Dgemv_hybrid") +
                          kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
    built_options.emplace(bias != nullptr ? "-DBIAS" : "");
    switch (activation_) {
    case NOOP: break;
    case RELU: built_options.emplace("-DUSE_RELU"); break;
    case RELUX: built_options.emplace("-DUSE_RELUX"); break;
    case TANH: built_options.emplace("-DUSE_TANH"); break;
    case SIGMOID: built_options.emplace("-DUSE_SIGMOID"); break;
    case LEAKYRELU: built_options.emplace("-DUSE_LEAKYRELU"); break;
    default: LOG(FATAL) << "Unknown activation type: " << activation_;
    }
    RETURN_IF_ERROR(
        runtime->BuildKernel("gemv", kernel_name, built_options, &kernel_));
  }

  OUT_OF_RANGE_INIT(kernel_);
  if (!has_kernel_init_) {
    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARGS(kernel_);
    SET_1D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(A->opencl_image()));
    if (has_opencl_image) {
      kernel_.setArg(idx++, *(B->opencl_image()));
    } else {
      kernel_.setArg(idx++, *(B->opencl_buffer()));
    }
    kernel_.setArg(idx++, *(bias->opencl_image()));
    kernel_.setArg(idx++, *(C->opencl_image()));
    kernel_.setArg(idx++, static_cast<int>(relux_max_limit_));
    kernel_.setArg(idx++, static_cast<int>(leakyrelu_coefficient_));
    kernel_.setArg(idx++, static_cast<int>(rows));
    kernel_.setArg(idx++, static_cast<int>(cols));
    has_kernel_init_ = true;
  }

  RETURN_IF_ERROR(Run1DKernel(runtime, kernel_, gws, lws, context->future()));
  OUT_OF_RANGE_VALIDATION;
  return VanState::SUCCEED;
}

} // namespace image
} // namespace opencl
} // namespace deepvan

#endif
