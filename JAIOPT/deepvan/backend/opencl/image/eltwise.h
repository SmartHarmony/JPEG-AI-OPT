#ifndef DEEPVAN_BACKEND_OPENCL_IMAGE_ELTWISE_H_
#define DEEPVAN_BACKEND_OPENCL_IMAGE_ELTWISE_H_

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "deepvan/backend/eltwise_op.h"
#include "deepvan/backend/opencl/eltwise.h"
#include "deepvan/backend/opencl/helper.h"
#include "deepvan/backend/opencl/image/image_tensor_debug.h"
#include "deepvan/core/op_context.h"
#include "deepvan/core/tensor.h"

namespace deepvan {
namespace opencl {
namespace image {

template <typename T>
class EltwiseKernel : public OpenCLEltwiseKernel {
public:
  explicit EltwiseKernel(const EltwiseType type,
                         const std::vector<float> &coeff,
                         const float scalar_input,
                         const int32_t scalar_input_index,
                         std::string name = "")
      : type_(type),
        coeff_(coeff),
        scalar_input_(scalar_input),
        scalar_input_index_(scalar_input_index),
        op_name_(name) {}
  VanState Compute(OpContext *context,
                   const Tensor *input0,
                   const Tensor *input1,
                   Tensor *output) override;

private:
  EltwiseType type_;
  std::vector<float> coeff_;
  float scalar_input_;
  int32_t scalar_input_index_;
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
  std::string op_name_;
};

template <typename T>
VanState EltwiseKernel<T>::Compute(OpContext *context,
                                   const Tensor *input0,
                                   const Tensor *input1,
                                   Tensor *output) {
  bool swapped = false;
  std::string input1_type = "";
  if (input1 == nullptr) {
    input1_type = "INPUT_SCALAR";
  } else {
    CONDITIONS(input0->dim_size() == input1->dim_size() ||
               input0->dim_size() == 1 || input1->dim_size() == 1)
        << "Inputs of Eltwise op must be same shape";
    CONDITIONS(type_ != EltwiseType::EQUAL)
        << "Eltwise op on GPU does not support EQUAL";
    // broadcast
    if (input0->size() != input1->size()) {
      if (input0->size() < input1->size()) {
        std::swap(input0, input1);
        swapped = true;
      }
      if (input1->dim_size() == 1 ||
          (input1->dim(0) == 1 && input1->dim(1) == 1 && input1->dim(2) == 1)) {
        // Tensor-Vector element wise
        if (input0->dim(3) == input1->dim(input1->dim_size() - 1)) {
          input1_type = "INPUT_VECTOR";
        } else if (input1->shape() == std::vector<index_t>{1,1,1,1}) {
          Tensor::MappingGuard input_mapper(input1);
          const T *input_ptr = input1->data<T>();
          scalar_input_ = input_ptr[0];
          input1_type = "INPUT_SCALAR";
        } else {
          LOG(FATAL) << "Inputs not match the broadcast logic, "
                     << MakeString(input0->shape()) << " vs "
                     << MakeString(input1->shape());
        }
      } else { // must be 4-D
        if (input0->dim(0) == input1->dim(0) && input1->dim(1) == 1 &&
            input1->dim(2) == 1 && input0->dim(3) == input1->dim(3)) {
          input1_type = "INPUT_BATCH_VECTOR";
        } else if (input0->dim(0) == input1->dim(0) &&
                   input0->dim(1) == input1->dim(1) &&
                   input0->dim(2) == input1->dim(2) && input1->dim(3) == 1) {
          // broadcast on channel dimension
          input1_type = "INPUT_TENSOR_BC_CHAN";
        } else {
          LOG(FATAL) << "Element-Wise op only support broadcast on"
                        " channel dimension:"
                        "Tensor-BatchVector(4D-[N,1,1,C]) "
                        "and Tensor-Tensor(4D-[N,H,W,1]). but got "
                     << MakeString(input0->shape()) << " vs "
                     << MakeString(input1->shape());
        }
      }
    }
  }

  if (scalar_input_index_ == 0) {
    swapped = !swapped;
  }

  std::vector<index_t> output_shape(4);
  output_shape[0] = input0->dim(0);
  output_shape[1] = input0->dim(1);
  output_shape[2] = input0->dim(2);
  output_shape[3] = input0->dim(3);

  std::vector<size_t> output_image_shape;
  OpenCLUtil::CalImage2DShape(
      output_shape, OpenCLBufferType::IN_OUT_CHANNEL, &output_image_shape, 0, context->model_type());
  RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));

  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t batch_height_pixels = batch * height;

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(batch_height_pixels)};

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  OUT_OF_RANGE_DEFINITION;
  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG;
    NON_UNIFORM_WG_CONFIG;
    auto dt = DataTypeToEnum<T>::value;
    std::string kernel_name = DEEPVAN_OBFUSCATE_SYMBOL("eltwise");
    built_options.emplace("-Deltwise=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
    built_options.emplace(MakeString("-DELTWISE_TYPE=", type_));
    if (!input1_type.empty()) {
      built_options.emplace("-D" + input1_type);
    }
    if (swapped)
      built_options.emplace("-DSWAPPED");
    if (channels % 4 != 0)
      built_options.emplace("-DNOT_DIVISIBLE_FOUR");
    if (!coeff_.empty())
      built_options.emplace("-DCOEFF_SUM");
    RETURN_IF_ERROR(
        runtime->BuildKernel("eltwise", kernel_name, built_options, &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }
  OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input0->shape())) {
    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARGS(kernel_);
    SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input0->opencl_image()));
    if (input1 == nullptr || input1_type == "INPUT_SCALAR" ) {
      kernel_.setArg(idx++, scalar_input_);
    } else {
      kernel_.setArg(idx++, *(input1->opencl_image()));
    }
    kernel_.setArg(idx++, static_cast<int32_t>(height));
    kernel_.setArg(idx++, static_cast<int32_t>(width));
    kernel_.setArg(idx++, static_cast<int32_t>(channels));
    if (!coeff_.empty()) {
      kernel_.setArg(idx++, coeff_[0]);
      kernel_.setArg(idx++, coeff_[1]);
    }
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = input0->shape();
  }

  const std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_);
  RETURN_IF_ERROR(Run3DKernel(runtime, kernel_, gws, lws, context->future()));
  OUT_OF_RANGE_VALIDATION;
  if (op_name_ == "['']") {
    std::vector<size_t> output_pitch = {0, 0};
    output->UnderlyingBuffer()->Map(&output_pitch);
    std::string message = "Eltwise " + op_name_;
    Show4DResult<T>(output, output_pitch, message.c_str(), 3, 3, false);
    output->UnderlyingBuffer()->UnMap();
  }
  // std::vector<size_t> output_pitch = {0, 0};
  // image::Write4DTensorToFile<float>(output, output_pitch, op_name_, true);
  return VanState::SUCCEED;
}

} // namespace image
} // namespace opencl
} // namespace deepvan

#endif // DEEPVAN_BACKEND_OPENCL_IMAGE_ELTWISE_H_
