#include "deepvan/backend/arm/fp32/deconv_2d_general.h"

namespace deepvan {
namespace arm {
namespace fp32 {

VanState Deconv2dGeneral::Compute(const OpContext *context,
                                  const Tensor *input,
                                  const Tensor *filter,
                                  const Tensor *output_shape,
                                  Tensor *output) {
  std::unique_ptr<Tensor> padded_out;
  std::vector<int> out_pad_size;
  ResizeOutAndPadOut(
      context, input, filter, output_shape, output, &out_pad_size, &padded_out);
  Tensor *out_tensor = output;
  if (padded_out != nullptr) {
    out_tensor = padded_out.get();
  }

  out_tensor->Clear();

  Tensor::MappingGuard input_mapper(input);
  Tensor::MappingGuard filter_mapper(filter);
  Tensor::MappingGuard output_mapper(out_tensor);

  auto input_data = input->data<float>();
  auto filter_data = filter->data<float>();
  auto padded_out_data = out_tensor->mutable_data<float>();

  auto &in_shape = input->shape();
  auto &out_shape = out_tensor->shape();

  const index_t out_height = out_shape[2];
  const index_t out_width = out_shape[3];
  const index_t in_height = in_shape[2];
  const index_t in_width = in_shape[3];
  const index_t out_img_size = out_height * out_width;
  const index_t in_img_size = in_height * in_width;
  const index_t kernel_h = filter->dim(2);
  const index_t kernel_w = filter->dim(3);

  const int kernel_size = static_cast<int>(kernel_h * kernel_w);
  std::vector<index_t> index_map(kernel_size, 0);
  for (index_t i = 0; i < kernel_h; ++i) {
    for (index_t j = 0; j < kernel_w; ++j) {
      index_map[i * kernel_w + j] = i * out_width + j;
    }
  }

  const index_t batch = in_shape[0];
  const index_t out_channels = out_shape[1];
  const index_t in_channels = in_shape[1];

#pragma omp parallel for collapse(2) schedule(runtime)
  for (int b = 0; b < batch; ++b) {
    for (int oc = 0; oc < out_channels; ++oc) {
      float *out_base =
          padded_out_data + (b * out_channels + oc) * out_img_size;
      for (int i = 0; i < in_height; ++i) {
        for (int j = 0; j < in_width; ++j) {
          const index_t out_offset =
              i * strides_[0] * out_width + j * strides_[1];
          for (int ic = 0; ic < in_channels; ++ic) {
            const index_t input_idx =
                (b * in_channels + ic) * in_img_size + i * in_width + j;
            const float val = input_data[input_idx];
            const index_t kernel_offset = (oc * in_channels + ic) * kernel_size;
            for (int k = 0; k < kernel_size; ++k) {
              const index_t out_idx = out_offset + index_map[k];
              const index_t kernel_idx = kernel_offset + k;
              out_base[out_idx] += val * filter_data[kernel_idx];
            }
          }
        }
      }
    }
  }

  UnPadOutput(*out_tensor, out_pad_size, output);

  return VanState::SUCCEED;
}

} // namespace fp32
} // namespace arm
} // namespace deepvan
