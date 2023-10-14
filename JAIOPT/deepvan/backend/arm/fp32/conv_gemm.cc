#include "deepvan/backend/arm/fp32/conv_gemm.h"
#include "deepvan/backend/common/conv_pool_2d_util.h"

namespace deepvan {
namespace arm {
namespace fp32 {

VanState Conv2dGEMM::Compute(const OpContext *context, const Tensor *input,
                             const Tensor *filter, Tensor *output) {
  auto in_shape = input->shape();
  auto filter_shape = filter->shape();
  std::vector<index_t> out_shape(4);
  auto input_data = input->data<float>();
  auto output_data = output->data<float>();
  std::vector<int> padding_size(2);
  CalcNCHWPaddingAndOutputSize(
      in_shape.data(), filter_shape.data(), dilations_.data(), strides_.data(),
      padding_type_, out_shape.data(), padding_size.data());
  RETURN_IF_ERROR(output->Resize(out_shape));
  // allocate enough buffer to hold the intermediate results
  const index_t data_col_size =
      im2col::GetIm2colBufferSize(out_shape.data(), filter_shape.data());
  auto scratch_buffer = context->device()->scratch_buffer();
  scratch_buffer->Rewind();
  // add gemm scratch size into here
  scratch_buffer->GrowSize(data_col_size);
  std::unique_ptr<Tensor> input_col = make_unique<Tensor>(
      scratch_buffer->Scratch(data_col_size), DataType::DT_FLOAT);
  auto input_data_col = input_col->mutable_data<float>();

  im2col::dense::DenseIm2colNCHW<float>(
      input_data, in_shape, filter_shape[2], filter_shape[3], strides_[0],
      strides_[1], padding_size[0], padding_size[1], dilations_[0],
      dilations_[1], out_shape, input_data_col);
  // RESULT = WEIGHT * INPUT
  const index_t batch = in_shape[0];
  const index_t lhs_rows = filter_shape[0];
  const index_t lhs_cols = filter_shape[1] * filter_shape[2] * filter_shape[3];
  const index_t rhs_rows = lhs_cols;
  const index_t rhs_cols = out_shape[2] * out_shape[3];
  return gemm_.Compute(context, filter, input_col.get(), batch, lhs_rows,
                       lhs_cols, rhs_rows, rhs_cols, false, false, false, false,
                       true, output);
}

} // namespace fp32
} // namespace arm
} // namespace deepvan