#ifndef DEEPVAN_BACKEND_COMMON_CONV_POOL_2D_UTIL_H_
#define DEEPVAN_BACKEND_COMMON_CONV_POOL_2D_UTIL_H_

#include "deepvan/core/tensor.h"

namespace deepvan {
enum Padding {
  VALID = 0,  // No padding
  SAME = 1,   // Pads with half the filter size (rounded down) on both sides
  FULL = 2,   // Pads with one less than the filter size on both sides
};

enum RoundType {
  FLOOR = 0,
  CEIL = 1,
};



void CalcPaddingAndOutputSize(const index_t *input_shape,
                              const DataFormat input_format,
                              const index_t *filter_shape,
                              const DataFormat filter_format,
                              const int *dilations,
                              const int *strides,
                              Padding padding,
                              index_t *output_shape,
                              int *padding_size);

void CalcNCHWPaddingAndOutputSize(const index_t *input_shape,
                                  const index_t *filter_shape,
                                  const int *dilations,
                                  const int *strides,
                                  Padding padding,
                                  index_t *output_shape,
                                  int *padding_size);

void CalcNHWCPaddingAndOutputSize(const index_t *input_shape,
                                  const index_t *filter_shape,
                                  const int *dilations,
                                  const int *strides,
                                  Padding padding,
                                  index_t *output_shape,
                                  int *padding_size);

void CalcOutputSize(const index_t *input_shape,
                    const DataFormat input_format,
                    const index_t *filter_shape,
                    const DataFormat filter_format,
                    const int *padding_size,
                    const int *dilations,
                    const int *strides,
                    const RoundType round_type,
                    index_t *output_shape);

void CalcOutputSize(const index_t *input_shape,   // NHWC
                    const index_t *filter_shape,  // OIHW
                    const int *padding_size,
                    const int *dilations,
                    const int *strides,
                    const RoundType round_type,
                    index_t *output_shape);

void CalcNCHWOutputSize(const index_t *input_shape,
                    const index_t *filter_shape,
                    const int *padding_size,
                    const int *dilations,
                    const int *strides,
                    const RoundType round_type,
                    index_t *output_shape);

void CalcNCHWInputShape(const index_t *output_shape,
                        const index_t *filter_shape,
                        const int *strides,
                        const int *dilations,
                        index_t *input_shape);

void CalPaddingSize(const index_t *input_shape,   // NCHW
                    const index_t *filter_shape,  // OIHW
                    const int *dilations,
                    const int *strides,
                    Padding padding,
                    int *padding_size);


VanState ConstructNCHWInputWithSpecificPadding(const Tensor *input,
                               const int pad_top, const int pad_bottom,
                               const int pad_left, const int pad_right,
                               Tensor *output_tensor);

VanState ConstructNCHWInputWithPadding(const Tensor *input,
                                   const int *paddings,
                                   Tensor *output_tensor,
                                   bool padding_same_value = false);

VanState ConstructNHWCInputWithPadding(const Tensor *input,
                                   const int *paddings,
                                   Tensor *output_tensor,
                                   bool padding_same_value = false);

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
                                    DataFormat data_format);

}  // namespace deepvan

#endif  // DEEPVAN_BACKEND_COMMON_CONV_POOL_2D_UTIL_H_
