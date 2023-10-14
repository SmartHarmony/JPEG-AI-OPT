#include "deepvan/backend/arm/fp32/conv_2d_1x1.h"

namespace deepvan {
namespace arm {
namespace fp32 {

VanState Conv2dK1x1::ComputeK1x1S1(const OpContext *context,
                                   const Tensor *input,
                                   const Tensor *filter,
                                   Tensor *output) {
  index_t batch = input->dim(0);
  index_t in_height = input->dim(2);
  index_t in_width = input->dim(3);
  index_t in_channels = input->dim(1);

  std::vector<index_t> output_shape;
  std::vector<int> in_pad_size;
  std::vector<int> out_pad_size;
  CalOutputShapeAndPadSize(
      input, filter, 1, 1, &output_shape, &in_pad_size, &out_pad_size);
  RETURN_IF_ERROR(output->Resize(output_shape));

  const index_t out_channels = output_shape[1];
  const index_t out_height = output_shape[2];
  const index_t out_width = output_shape[3];
  const index_t padded_in_height = in_height + in_pad_size[0] + in_pad_size[1];
  const index_t padded_in_width = in_width + in_pad_size[2] + in_pad_size[3];

  // pad input and transform input
  const bool is_in_padded =
      in_height != padded_in_height || in_width != padded_in_width;
  auto scratch_buffer = context->device()->scratch_buffer();
  const index_t padded_in_size =
      is_in_padded ? PadAlignSize(sizeof(float) * batch * in_channels *
                                  padded_in_height * padded_in_width)
                   : 0;
  const index_t pack_filter_size =
      PadAlignSize(sizeof(float) * out_channels * in_channels);
  const index_t pack_input_size = PadAlignSize(
      sizeof(float) * in_channels * padded_in_height * padded_in_width);
  const index_t pack_output_size = PadAlignSize(
      sizeof(float) * out_channels * padded_in_height * padded_in_width);

  const index_t gemm_pack_size =
      pack_filter_size + pack_input_size + pack_output_size;

  scratch_buffer->Rewind();
  scratch_buffer->GrowSize(padded_in_size + gemm_pack_size);

  const Tensor *padded_in = input;
  Tensor tmp_padded_in(scratch_buffer->Scratch(padded_in_size),
                       DataType::DT_FLOAT);
  if (is_in_padded) {
    tmp_padded_in.Resize(
        {batch, in_channels, padded_in_height, padded_in_width});
    PadInput(*input, in_pad_size[0], in_pad_size[2], &tmp_padded_in);
    padded_in = &tmp_padded_in;
  }

  return gemm_.Compute(context,
                       filter,
                       padded_in,
                       batch,
                       out_channels,
                       in_channels,
                       in_channels,
                       out_height * out_width,
                       false,
                       false,
                       false,
                       false,
                       true,
                       output);
}

VanState Conv2dK1x1::ComputeK1x1S2(const OpContext *context,
                                   const Tensor *input,
                                   const Tensor *filter,
                                   Tensor *output) {
  index_t batch = input->dim(0);
  index_t in_height = input->dim(2);
  index_t in_width = input->dim(3);
  index_t in_channels = input->dim(1);

  std::vector<index_t> output_shape;
  std::vector<int> in_pad_size;
  std::vector<int> out_pad_size;
  CalOutputShapeAndPadSize(
      input, filter, 1, 1, &output_shape, &in_pad_size, &out_pad_size);
  RETURN_IF_ERROR(output->Resize(output_shape));

  const index_t out_channels = output_shape[1];
  const index_t out_height = output_shape[2];
  const index_t out_width = output_shape[3];
  const index_t stride_in_height = (in_height + 1) >> 1;
  const index_t stride_in_width = (in_width + 1) >> 1;

  auto scratch_buffer = context->device()->scratch_buffer();
  const index_t stride_in_size = PadAlignSize(
      sizeof(float) * batch * in_channels * stride_in_height * stride_in_width);
  const index_t pack_filter_size =
      PadAlignSize(sizeof(float) * out_channels * in_channels);
  const index_t pack_input_size = PadAlignSize(
      sizeof(float) * in_channels * stride_in_height * stride_in_width);
  const index_t pack_output_size = PadAlignSize(
      sizeof(float) * out_channels * stride_in_height * stride_in_width);

  const index_t gemm_pack_size =
      pack_filter_size + pack_input_size + pack_output_size;

  scratch_buffer->Rewind();
  scratch_buffer->GrowSize(stride_in_size + gemm_pack_size);

  const Tensor *stride_in = input;
  Tensor tmp_padded_in(scratch_buffer->Scratch(stride_in_size),
                       DataType::DT_FLOAT);
  tmp_padded_in.Resize({batch, in_channels, stride_in_height, stride_in_width});
  StrideInput(*input, &tmp_padded_in);
  stride_in = &tmp_padded_in;

  return gemm_.Compute(context,
                       filter,
                       stride_in,
                       batch,
                       out_channels,
                       in_channels,
                       in_channels,
                       out_height * out_width,
                       false,
                       false,
                       false,
                       false,
                       true,
                       output);
}

VanState Conv2dK1x1::Compute(const OpContext *context,
                             const Tensor *input,
                             const Tensor *filter,
                             Tensor *output) {
  VanState status;
  if (strides_[0] == 1 && strides_[1] == 1) {
    status = ComputeK1x1S1(context, input, filter, output);
  } else {
    status = ComputeK1x1S2(context, input, filter, output);
  }
  return status;
}

} // namespace fp32
} // namespace arm
} // namespace deepvan
