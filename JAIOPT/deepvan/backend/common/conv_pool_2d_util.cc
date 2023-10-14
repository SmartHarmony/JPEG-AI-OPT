#include "deepvan/backend/common/conv_pool_2d_util.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace deepvan {

void CalcPaddingAndOutputSize(const index_t *input_shape,
                              const DataFormat input_format,
                              const index_t *filter_shape,
                              const DataFormat filter_format,
                              const int *dilations,
                              const int *strides,
                              Padding padding,
                              index_t *output_shape,
                              int *padding_size) {
  CONDITIONS(dilations[0] > 0 && dilations[1] > 0,
             "Invalid dilations, must >= 1");
  CONDITIONS((dilations[0] == 1 || strides[0] == 1) &&
                 (dilations[1] == 1 || strides[1] == 1),
             "If dilations > 1, strides should be 1");
  CONDITIONS_NOTNULL(output_shape);
  CONDITIONS_NOTNULL(padding_size);

  index_t input_height = 0, input_width = 0;
  index_t kernel_height = 0, kernel_width = 0;
  if (input_format == NCHW) {
    input_height = input_shape[2];
    input_width = input_shape[3];
  } else if (input_format == NHWC) {
    input_height = input_shape[1];
    input_width = input_shape[2];
  } else {
    STUB;
  }
  if (filter_format == OIHW) {
    kernel_height = filter_shape[2];
    kernel_width = filter_shape[3];
  } else if (filter_format == OHWI) {
    kernel_height = filter_shape[1];
    kernel_width = filter_shape[2];
  } else {
    STUB;
  }
  /*
   * Convlution/pooling arithmetic:
   * o = (i + 2 * p - k - (k - 1) * (d - 1)) / s + 1
   * For details, see https://arxiv.org/pdf/1603.07285.pdf or
   * http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
   */
  padding_size[0] = 0;
  padding_size[1] = 0;
  index_t output_height = 0, output_width = 0;
  index_t output_channels = filter_shape[0];
  index_t k_extent_height = (kernel_height - 1) * dilations[0] + 1;
  index_t k_extent_width = (kernel_width - 1) * dilations[1] + 1;

  switch (padding) {
  case VALID:
    output_height = (input_height - k_extent_height) / strides[0] + 1;
    output_width = (input_width - k_extent_width) / strides[1] + 1;
    break;
  case SAME:
    output_height = (input_height - 1) / strides[0] + 1;
    output_width = (input_width - 1) / strides[1] + 1;
    break;
  case FULL:
    output_height = (input_height + k_extent_height - 2) / strides[0] + 1;
    output_width = (input_width + k_extent_width - 2) / strides[1] + 1;
    break;
  default: CONDITIONS(false, "Unsupported padding type: ", padding);
  }

  // Note: TensorFlow may padded one more on the right/bottom side
  // may be it's better to also truncate the left/top to
  // utilize the more centered features. We need to benchmark
  // based on the model accuracy.

  padding_size[0] = std::max<int>(
      0, (output_height - 1) * strides[0] + k_extent_height - input_height);
  padding_size[1] = std::max<int>(
      0, (output_width - 1) * strides[1] + k_extent_width - input_width);

  output_shape[0] = input_shape[0];
  if (input_format == NCHW) {
    output_shape[1] = output_channels;
    output_shape[2] = output_height;
    output_shape[3] = output_width;
  } else if (input_format == NHWC) {
    output_shape[1] = output_height;
    output_shape[2] = output_width;
    output_shape[3] = output_channels;
  } else {
    STUB;
  }
}

void CalcNCHWPaddingAndOutputSize(const index_t *input_shape,  // NCHW
                                  const index_t *filter_shape, // OIHW
                                  const int *dilations,
                                  const int *strides,
                                  Padding padding,
                                  index_t *output_shape,
                                  int *padding_size) {
  CalcPaddingAndOutputSize(input_shape,
                           NCHW,
                           filter_shape,
                           OIHW,
                           dilations,
                           strides,
                           padding,
                           output_shape,
                           padding_size);
}

void CalcNHWCPaddingAndOutputSize(const index_t *input_shape,  // NHWC
                                  const index_t *filter_shape, // OIHW
                                  const int *dilations,
                                  const int *strides,
                                  Padding padding,
                                  index_t *output_shape,
                                  int *padding_size) {
  CalcPaddingAndOutputSize(input_shape,
                           NHWC,
                           filter_shape,
                           OIHW,
                           dilations,
                           strides,
                           padding,
                           output_shape,
                           padding_size);
}

void CalcOutputSize(const index_t *input_shape,
                    const DataFormat input_format,
                    const index_t *filter_shape,
                    const DataFormat filter_format,
                    const int *padding_size,
                    const int *dilations,
                    const int *strides,
                    const RoundType round_type,
                    index_t *output_shape) {
  CONDITIONS(dilations[0] > 0 && dilations[1] > 0,
             "Invalid dilations, must >= 1");
  CONDITIONS((dilations[0] == 1 || strides[0] == 1) &&
                 (dilations[1] == 1 || strides[1] == 1),
             "If dilations > 1, strides should be 1");
  CONDITIONS_NOTNULL(output_shape);
  CONDITIONS_NOTNULL(padding_size);

  index_t input_height = 0, input_width = 0;
  index_t kernel_height = 0, kernel_width = 0;
  if (input_format == NCHW) {
    input_height = input_shape[2];
    input_width = input_shape[3];
  } else if (input_format == NHWC) {
    input_height = input_shape[1];
    input_width = input_shape[2];
  } else {
    STUB;
  }
  if (filter_format == OIHW) {
    kernel_height = filter_shape[2];
    kernel_width = filter_shape[3];
  } else if (filter_format == OHWI) {
    kernel_height = filter_shape[1];
    kernel_width = filter_shape[2];
  } else {
    STUB;
  }
  /*
   * Convlution/pooling arithmetic:
   * o = (i + 2 * p - k - (k - 1) * (d - 1)) / s + 1
   * For details, see https://arxiv.org/pdf/1603.07285.pdf or
   * http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
   */
  index_t output_height = 0, output_width = 0;
  index_t output_channels = filter_shape[0];

  if (round_type == FLOOR) {
    output_height = static_cast<index_t>(
        std::floor(1.0 *
                   (input_height + padding_size[0] - kernel_height -
                    (kernel_height - 1) * (dilations[0] - 1)) /
                   strides[0]) +
        1);
    output_width = static_cast<index_t>(
        std::floor(1.0 *
                   (input_width + padding_size[1] - kernel_width -
                    (kernel_width - 1) * (dilations[1] - 1)) /
                   strides[1]) +
        1);
  } else {
    output_height = static_cast<index_t>(
        std::ceil(1.0 *
                  (input_height + padding_size[0] - kernel_height -
                   (kernel_height - 1) * (dilations[0] - 1)) /
                  strides[0]) +
        1);
    output_width = static_cast<index_t>(
        std::ceil(1.0 *
                  (input_width + padding_size[1] - kernel_width -
                   (kernel_width - 1) * (dilations[1] - 1)) /
                  strides[1]) +
        1);
  }

  output_shape[0] = input_shape[0];
  if (input_format == NCHW) {
    output_shape[1] = output_channels;
    output_shape[2] = output_height;
    output_shape[3] = output_width;
  } else if (input_format == NHWC) {
    output_shape[1] = output_height;
    output_shape[2] = output_width;
    output_shape[3] = output_channels;
  } else {
    STUB;
  }
}

void CalcNCHWInputShape(const index_t *output_shape,
                        const index_t *filter_shape,
                        const int *strides,
                        const int *dilations,
                        index_t *input_shape) {
  CONDITIONS_NOTNULL(input_shape);
  input_shape[0] = output_shape[0];
  input_shape[1] = filter_shape[1];
  input_shape[2] = (output_shape[2] - 1) * strides[0] +
                   (filter_shape[2] - 1) * dilations[0] + 1;
  input_shape[3] = (output_shape[3] - 1) * strides[1] +
                   (filter_shape[3] - 1) * dilations[1] + 1;
}

void CalcOutputSize(const index_t *input_shape,  // NHWC
                    const index_t *filter_shape, // OIHW
                    const int *padding_size,
                    const int *dilations,
                    const int *strides,
                    const RoundType round_type,
                    index_t *output_shape) {
  CalcOutputSize(input_shape,
                 NHWC,
                 filter_shape,
                 OIHW,
                 padding_size,
                 dilations,
                 strides,
                 round_type,
                 output_shape);
}

void CalcNCHWOutputSize(const index_t *input_shape,  // NCHW
                        const index_t *filter_shape, // OIHW
                        const int *padding_size,
                        const int *dilations,
                        const int *strides,
                        const RoundType round_type,
                        index_t *output_shape) {
  CalcOutputSize(input_shape,
                 NCHW,
                 filter_shape,
                 OIHW,
                 padding_size,
                 dilations,
                 strides,
                 round_type,
                 output_shape);
}

void CalPaddingSize(const index_t *input_shape,  // NCHW
                    const index_t *filter_shape, // OIHW
                    const int *strides,
                    const int *dilations,
                    Padding padding,
                    int *padding_size) {
  CONDITIONS(dilations[0] > 0 && dilations[1] > 0,
             "Invalid dilations, must >= 1");
  CONDITIONS((dilations[0] == 1 || strides[0] == 1) &&
                 (dilations[1] == 1 || strides[1] == 1),
             "If dilations > 1, strides should be 1");
  CONDITIONS_NOTNULL(padding_size);

  index_t output_height = 0, output_width = 0;
  index_t k_extent_height = (filter_shape[2] - 1) * dilations[0] + 1;
  index_t k_extent_width = (filter_shape[3] - 1) * dilations[1] + 1;

  switch (padding) {
  case VALID:
    output_height = (input_shape[2] - k_extent_height) / strides[0] + 1;
    output_width = (input_shape[3] - k_extent_width) / strides[1] + 1;
    break;
  case SAME:
    output_height = (input_shape[2] - 1) / strides[0] + 1;
    output_width = (input_shape[3] - 1) / strides[1] + 1;
    break;
  case FULL:
    output_height = (input_shape[2] + k_extent_height - 2) / strides[0] + 1;
    output_width = (input_shape[3] + k_extent_width - 2) / strides[1] + 1;
    break;
  default: CONDITIONS(false, "Unsupported padding type: ", padding);
  }

  // Note: TensorFlow may padded one more on the right/bottom side
  // TODO(@vgod): may be it's better to also truncate the left/top to
  // utilize the more centered features. We need to benchmark
  // based on the model accuracy.
  padding_size[0] = std::max<int>(
      0, (output_height - 1) * strides[0] + k_extent_height - input_shape[2]);
  padding_size[1] = std::max<int>(
      0, (output_width - 1) * strides[1] + k_extent_width - input_shape[3]);
}

VanState ConstructNCHWInputWithPadding(const Tensor *input_tensor,
                                       const int *paddings,
                                       Tensor *output_tensor,
                                       bool padding_same_value) {
  Tensor::MappingGuard input_mapper(input_tensor);
  const float *input = input_tensor->data<float>();
  const index_t *input_shape = input_tensor->shape().data();

  index_t batch = input_shape[0];
  index_t channels = input_shape[1];
  index_t height = input_shape[2];
  index_t width = input_shape[3];

  std::vector<index_t> output_shape(
      {batch, channels, paddings[0] + height, paddings[1] + width});

  const index_t output_width = output_shape[3];
  const int padded_top = paddings[0] / 2;
  const int padded_left = paddings[1] / 2;

  RETURN_IF_ERROR(output_tensor->Resize(output_shape));

  Tensor::MappingGuard padded_output_mapper(output_tensor);
  float *output_data = output_tensor->mutable_data<float>();
  memset(output_data, 0, output_tensor->size() * sizeof(float));

  // Skip the padded top rows
  if (padding_same_value) {
#define DEEPVAN_COPY_INPUT                                                     \
  std::fill(output_data, output_data + padded_left, input[0]);                 \
  output_data += padded_left;                                                  \
  memcpy(output_data, input, width * sizeof(float));                           \
  output_data += width;                                                        \
  std::fill(output_data, output_data + padded_right, input[width - 1]);        \
  output_data += padded_right;

    const int padded_bottom = paddings[0] - padded_top;
    const int padded_right = paddings[1] - padded_left;

    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        for (int k = 0; k < padded_top; ++k) {
          DEEPVAN_COPY_INPUT;
        }
        for (int k = 0; k < height; ++k) {
          DEEPVAN_COPY_INPUT;
          input += width;
        }
        input -= width;
        for (int k = 0; k < padded_bottom; ++k) {
          DEEPVAN_COPY_INPUT;
        }
        input += width;
      }
    }
#undef DEEPVAN_COPY_INPUT
  } else {
    output_data += padded_top * output_width;
    for (int i = 0; i < batch; ++i) {
      for (int j = 0; j < channels; ++j) {
        for (int k = 0; k < height; ++k) {
          memcpy(output_data + padded_left, input, width * sizeof(float));
          input += width;
          output_data += output_width;
        }
        // Skip the padded bottom in this channel and top in the next channel
        output_data += paddings[0] * output_width;
      }
    }
  }

  return VanState::SUCCEED;
}

VanState ConstructNCHWInputWithSpecificPadding(const Tensor *input_tensor,
                                               const int pad_top,
                                               const int pad_bottom,
                                               const int pad_left,
                                               const int pad_right,
                                               Tensor *output_tensor) {
  const float *input = input_tensor->data<float>();
  const index_t *input_shape = input_tensor->shape().data();

  index_t batch = input_shape[0];
  index_t channels = input_shape[1];
  index_t height = input_shape[2];
  index_t width = input_shape[3];

  const int pad_height = pad_top + pad_bottom;
  const int pad_width = pad_left + pad_right;
  std::vector<index_t> output_shape(
      {batch, channels, height + pad_height, width + pad_width});
  RETURN_IF_ERROR(output_tensor->Resize(output_shape));
  output_tensor->Clear();
  Tensor::MappingGuard padded_output_mapper(output_tensor);
  float *output_data = output_tensor->mutable_data<float>();

  const index_t output_height = output_shape[2];
  const index_t output_width = output_shape[3];
  const index_t in_image_size = height * width;
  const index_t out_image_size = output_height * output_width;
  const index_t in_batch_size = channels * in_image_size;
  const index_t out_batch_size = channels * out_image_size;

#pragma omp parallel for collapse(2) schedule(runtime)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < height; ++k) {
        memcpy(output_data + i * out_batch_size + j * out_image_size +
                   (pad_top + k) * output_width + pad_left,
               input + i * in_batch_size + j * in_image_size + k * width,
               width * sizeof(float));
      }
      // Skip the padded bottom in this channel and top in the next channel
    }
  }

  return VanState::SUCCEED;
}

VanState ConstructNHWCInputWithPadding(const Tensor *input_tensor,
                                       const int *paddings,
                                       Tensor *output_tensor,
                                       bool padding_same_value) {
  Tensor::MappingGuard input_mapper(input_tensor);
  const float *input = input_tensor->data<float>();
  const index_t *input_shape = input_tensor->shape().data();

  index_t batch = input_shape[0];
  index_t height = input_shape[1];
  index_t width = input_shape[2];
  index_t channels = input_shape[3];

  std::vector<index_t> output_shape(
      {batch, paddings[0] + height, paddings[1] + width, channels});

  const int output_height = output_shape[1];
  const int output_width = output_shape[2];
  const int padded_top = paddings[0] / 2;
  const int padded_left = paddings[1] / 2;

  RETURN_IF_ERROR(output_tensor->Resize(output_shape));

  Tensor::MappingGuard padded_output_mapper(output_tensor);
  float *output_data = output_tensor->mutable_data<float>();
  memset(output_data, 0, output_tensor->size() * sizeof(float));

  // Skip the padded top rows
  if (padding_same_value) {
    LOG(FATAL) << "Not implemented";
  } else {
#pragma omp parallel for collapse(3) schedule(runtime)
    for (int n = 0; n < batch; ++n) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          const float *input_ptr =
              input + ((n * height + h) * width + w) * channels;
          float *output_ptr =
              output_data +
              ((n * output_height + h + padded_top) * output_width + w +
               padded_left) *
                  channels;
          memcpy(output_ptr, input_ptr, channels * sizeof(float));
        }
      }
    }
  }

  return VanState::SUCCEED;
}

void CalcDeconvShape_TF(const std::vector<index_t> &input_shape,
                        const std::vector<index_t> &filter_shape,
                        const std::vector<index_t> &output_shape,
                        const std::vector<int> &strides,
                        Padding padding_type,
                        const int group,
                        std::vector<int> *in_pad_size,
                        std::vector<int> *out_pad_size,
                        std::vector<index_t> *padded_out_shape,
                        DataFormat data_format) {
  const index_t in_height =
      data_format == DataFormat::NCHW ? input_shape[2] : input_shape[1];
  const index_t in_width =
      data_format == DataFormat::NCHW ? input_shape[3] : input_shape[2];

  const index_t out_height =
      data_format == DataFormat::NCHW ? output_shape[2] : output_shape[1];
  const index_t out_width =
      data_format == DataFormat::NCHW ? output_shape[3] : output_shape[2];

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

  if (in_pad_size != nullptr) {
    const int p_h =
        static_cast<int>(out_height + kernel_h - 1 - extended_in_height);
    const int p_w =
        static_cast<int>(out_width + kernel_w - 1 - extended_in_width);
    in_pad_size->resize(2);
    (*in_pad_size)[0] = std::max<int>(0, p_h);
    (*in_pad_size)[1] = std::max<int>(0, p_w);
  }

  if (out_pad_size != nullptr) {
    const int o_p_h = static_cast<int>(padded_out_height - out_height);
    const int o_p_w = static_cast<int>(padded_out_width - out_width);
    out_pad_size->resize(2);
    (*out_pad_size)[0] = std::max<int>(0, o_p_h);
    (*out_pad_size)[1] = std::max<int>(0, o_p_w);
  }

  if (padded_out_shape != nullptr) {
    index_t output_channel = filter_shape[0] * group;
    padded_out_shape->resize(4);
    (*padded_out_shape)[0] = output_shape[0];
    (*padded_out_shape)[1] =
        data_format == DataFormat::NCHW ? output_channel : padded_out_height;
    (*padded_out_shape)[2] =
        data_format == DataFormat::NCHW ? padded_out_height : padded_out_width;
    (*padded_out_shape)[3] =
        data_format == DataFormat::NCHW ? padded_out_width : output_channel;
  }
}

void CalcDeconvShape_Caffe(const std::vector<index_t> &input_shape,
                           const std::vector<index_t> &filter_shape,
                           const std::vector<int> &strides,
                           const std::vector<int> &out_pad_size,
                           const int group,
                           std::vector<index_t> *out_shape,
                           std::vector<int> *in_pad_size,
                           std::vector<index_t> *padded_out_shape,
                           DataFormat data_format) {
  const index_t in_height =
      data_format == DataFormat::NCHW ? input_shape[2] : input_shape[1];
  const index_t in_width =
      data_format == DataFormat::NCHW ? input_shape[3] : input_shape[2];

  const index_t output_channel = filter_shape[0] * group;

  const index_t kernel_h = filter_shape[2];
  const index_t kernel_w = filter_shape[3];

  index_t padded_out_height = (in_height - 1) * strides[0] + kernel_h;
  index_t padded_out_width = (in_width - 1) * strides[1] + kernel_w;

  if (in_pad_size != nullptr) {
    in_pad_size->resize(2);
    (*in_pad_size)[0] = static_cast<int>((kernel_h - 1) * 2 - out_pad_size[0]);
    (*in_pad_size)[1] = static_cast<int>((kernel_w - 1) * 2 - out_pad_size[1]);
    (*in_pad_size)[0] = std::max<int>(0, (*in_pad_size)[0]);
    (*in_pad_size)[1] = std::max<int>(0, (*in_pad_size)[1]);
  }

  if (padded_out_shape != nullptr) {
    padded_out_shape->resize(4);
    (*padded_out_shape)[0] = input_shape[0];
    (*padded_out_shape)[1] =
        data_format == DataFormat::NCHW ? output_channel : padded_out_height;
    (*padded_out_shape)[2] =
        data_format == DataFormat::NCHW ? padded_out_height : padded_out_width;
    (*padded_out_shape)[3] =
        data_format == DataFormat::NCHW ? padded_out_width : output_channel;
  }

  if (out_shape != nullptr) {
    index_t out_height = padded_out_height - out_pad_size[0];
    index_t out_width = padded_out_width - out_pad_size[1];
    out_shape->resize(4);
    (*out_shape)[0] = input_shape[0];
    (*out_shape)[1] =
        data_format == DataFormat::NCHW ? output_channel : out_height;
    (*out_shape)[2] = data_format == DataFormat::NCHW ? out_height : out_width;
    (*out_shape)[3] =
        data_format == DataFormat::NCHW ? out_width : output_channel;
  }
}

void CalDeconvOutputShapeAndPadSize(const std::vector<index_t> &input_shape,
                                    const std::vector<index_t> &filter_shape,
                                    const std::vector<int> &strides,
                                    Padding padding_type,
                                    const std::vector<int> &paddings,
                                    int group,
                                    std::vector<index_t> *output_shape,
                                    std::vector<int> *in_pad_size,
                                    std::vector<int> *out_pad_size,
                                    std::vector<index_t> *padded_out_shape,
                                    FrameworkType framework_type,
                                    DataFormat data_format) {
  if (framework_type == FrameworkType::TENSORFLOW) {
    CONDITIONS(output_shape->size() == 4,
               "deconv output shape shoud be 4-dims");
    std::vector<index_t> &out_shape = *output_shape;
    if (data_format == DataFormat::NCHW) {
      const index_t t = out_shape[1];
      out_shape[1] = out_shape[3];
      out_shape[3] = out_shape[2];
      out_shape[2] = t;
    }

    CalcDeconvShape_TF(input_shape,
                       filter_shape,
                       *output_shape,
                       strides,
                       padding_type,
                       group,
                       in_pad_size,
                       out_pad_size,
                       padded_out_shape,
                       data_format);
  } else { // caffe
    if (!paddings.empty())
      *out_pad_size = paddings;
    CalcDeconvShape_Caffe(input_shape,
                          filter_shape,
                          strides,
                          *out_pad_size,
                          group,
                          output_shape,
                          in_pad_size,
                          padded_out_shape,
                          data_format);
  }
}

} // namespace deepvan
