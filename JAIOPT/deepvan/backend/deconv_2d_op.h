#ifndef DEEPVAN_BACKEND_DECONV_2D_OP_H_
#define DEEPVAN_BACKEND_DECONV_2D_OP_H_

#include <algorithm>
#include <string>
#include <vector>

#include "deepvan/backend/activation_op.h"
#include "deepvan/backend/common/conv_pool_2d_util.h"
#include "deepvan/core/operator.h"
#include "deepvan/core/types.h"
namespace deepvan {

class Deconv2dOpBase : public Operation {
public:
  explicit Deconv2dOpBase(OpConstructContext *context)
      : Operation(context),
        strides_(Operation::GetRepeatedArgs<int>("strides")),
        padding_type_(static_cast<Padding>(
            Operation::GetOptionalArg<int>("padding", static_cast<int>(SAME)))),
        paddings_(Operation::GetRepeatedArgs<int>("padding_values")),
        output_padding_(Operation::GetRepeatedArgs<int>("output_padding")),
        group_(Operation::GetOptionalArg<int>("group", 1)),
        model_type_(static_cast<FrameworkType>(
            Operation::GetOptionalArg<int>("framework_type", 0))),
        activation_(StringToActivationType(
            Operation::GetOptionalArg<std::string>("activation", "NOOP"))),
        relux_max_limit_(Operation::GetOptionalArg<float>("max_limit", 0.0f)),
        leakyrelu_coefficient_(
            Operation::GetOptionalArg<float>("leakyrelu_coefficient", 0.0f)) {}

  static void CalcDeconvShape_Caffe(const index_t *input_shape,  // NHWC
                                    const index_t *filter_shape, // OIHW
                                    const int *strides,
                                    const int *out_paddings,
                                    const int *output_padding,
                                    const int group,
                                    int *in_paddings,
                                    index_t *out_shape,
                                    index_t *padded_out_shape,
                                    const bool isNCHW = false) {
    CONDITIONS_NOTNULL(out_paddings);
    CONDITIONS_NOTNULL(input_shape);
    CONDITIONS_NOTNULL(filter_shape);
    CONDITIONS_NOTNULL(strides);

    const index_t in_height = isNCHW ? input_shape[2] : input_shape[1];
    const index_t in_width = isNCHW ? input_shape[3] : input_shape[2];

    const index_t output_channel = filter_shape[0] * group;

    const index_t kernel_h = filter_shape[2];
    const index_t kernel_w = filter_shape[3];

    index_t padded_out_height = (in_height - 1) * strides[0] + kernel_h;
    index_t padded_out_width = (in_width - 1) * strides[1] + kernel_w;

    if (in_paddings != nullptr) {
      in_paddings[0] = static_cast<int>((kernel_h - 1) * 2 - out_paddings[0]);
      in_paddings[1] = static_cast<int>((kernel_w - 1) * 2 - out_paddings[1]);
      in_paddings[0] = std::max<int>(0, in_paddings[0]);
      in_paddings[1] = std::max<int>(0, in_paddings[1]);
    }

    if (padded_out_shape != nullptr) {
      padded_out_shape[0] = input_shape[0];
      padded_out_shape[1] = isNCHW ? output_channel : padded_out_height;
      padded_out_shape[2] = isNCHW ? padded_out_height : padded_out_width;
      padded_out_shape[3] = isNCHW ? padded_out_width : output_channel;
    }

    if (out_shape != nullptr) {
      index_t out_height = padded_out_height - out_paddings[0];
      index_t out_width = padded_out_width - out_paddings[1];
      out_shape[0] = input_shape[0];
      out_shape[1] = isNCHW ? output_channel : out_height;
      out_shape[2] = isNCHW ? out_height : out_width;
      out_shape[3] = isNCHW ? out_width : output_channel;
    }

    if (output_padding != nullptr) {
      if (isNCHW) {
        out_shape[2] += output_padding[0];
        out_shape[3] += output_padding[1];
      } else {
        out_shape[1] += output_padding[0];
        out_shape[2] += output_padding[1];
      }
    }
  }

  static void CalcDeconvShape_TF(const index_t *input_shape,  // NHWC
                                 const index_t *filter_shape, // OIHW
                                 const index_t *output_shape,
                                 const int *strides,
                                 const int group,
                                 Padding padding_type,
                                 int *in_paddings,
                                 int *out_paddings,
                                 index_t *padded_out_shape,
                                 const bool isNCHW = false) {
    CONDITIONS_NOTNULL(output_shape);
    CONDITIONS_NOTNULL(input_shape);
    CONDITIONS_NOTNULL(filter_shape);
    CONDITIONS_NOTNULL(strides);

    const index_t in_height = isNCHW ? input_shape[2] : input_shape[1];
    const index_t in_width = isNCHW ? input_shape[3] : input_shape[2];

    const index_t out_height = isNCHW ? output_shape[2] : output_shape[1];
    const index_t out_width = isNCHW ? output_shape[3] : output_shape[2];

    const index_t extended_in_height = (in_height - 1) * strides[0] + 1;
    const index_t extended_in_width = (in_width - 1) * strides[1] + 1;

    const index_t kernel_h = filter_shape[2];
    const index_t kernel_w = filter_shape[3];

    index_t expected_input_height = 0, expected_input_width = 0;

    switch (padding_type) {
    case VALID:
      expected_input_height = (out_height - kernel_h + strides[0]) / strides[0];
      expected_input_width = (out_width - kernel_w + strides[1]) / strides[1];
      break;
    case SAME:
      expected_input_height = (out_height + strides[0] - 1) / strides[0];
      expected_input_width = (out_width + strides[1] - 1) / strides[1];
      break;
    default: CONDITIONS(false, "Unsupported padding type: ", padding_type);
    }

    CONDITIONS(expected_input_height == in_height,
               expected_input_height,
               "!=",
               in_height);
    CONDITIONS(
        expected_input_width == in_width, expected_input_width, "!=", in_width);

    const index_t padded_out_height = (in_height - 1) * strides[0] + kernel_h;
    const index_t padded_out_width = (in_width - 1) * strides[1] + kernel_w;

    if (in_paddings != nullptr) {
      const int p_h =
          static_cast<int>(out_height + kernel_h - 1 - extended_in_height);
      const int p_w =
          static_cast<int>(out_width + kernel_w - 1 - extended_in_width);
      in_paddings[0] = std::max<int>(0, p_h);
      in_paddings[1] = std::max<int>(0, p_w);
    }

    if (out_paddings != nullptr) {
      const int o_p_h = static_cast<int>(padded_out_height - out_height);
      const int o_p_w = static_cast<int>(padded_out_width - out_width);
      out_paddings[0] = std::max<int>(0, o_p_h);
      out_paddings[1] = std::max<int>(0, o_p_w);
    }

    if (padded_out_shape != nullptr) {
      index_t output_channel = filter_shape[0] * group;
      padded_out_shape[0] = output_shape[0];
      padded_out_shape[1] = isNCHW ? output_channel : padded_out_height;
      padded_out_shape[2] = isNCHW ? padded_out_height : padded_out_width;
      padded_out_shape[3] = isNCHW ? padded_out_width : output_channel;
    }
  }

protected:
  std::vector<int> strides_; // [stride_h, stride_w]
  const Padding padding_type_;
  std::vector<int> paddings_;
  std::vector<int> output_padding_;
  const int group_;
  const FrameworkType model_type_;
  const ActivationType activation_;
  const float relux_max_limit_;
  const float leakyrelu_coefficient_;
};

template <typename T>
void CropPadOut(const T *input,
                const index_t *in_shape,
                const index_t *out_shape,
                const index_t pad_h,
                const index_t pad_w,
                T *output) {
  const index_t batch = in_shape[0];
  const index_t channel = in_shape[1];
  const index_t in_height = in_shape[2];
  const index_t in_width = in_shape[3];

  const index_t out_height = out_shape[2];
  const index_t out_width = out_shape[3];
#pragma omp parallel for collapse(3)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channel; ++j) {
      for (int k = 0; k < out_height; ++k) {
        const T *input_base =
            input + ((i * channel + j) * in_height + (k + pad_h)) * in_width;
        T *output_base =
            output + ((i * channel + j) * out_height + k) * out_width;
        memcpy(output_base, input_base + pad_w, out_width * sizeof(T));
      }
    }
  }
}

} // namespace deepvan

#endif // DEEPVAN_BACKEND_DECONV_OP_2D_H_
