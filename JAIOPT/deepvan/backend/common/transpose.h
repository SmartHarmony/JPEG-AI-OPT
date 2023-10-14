#ifndef DEEPVAN_BACKEND_COMMON_TRANSPOSE_H_
#define DEEPVAN_BACKEND_COMMON_TRANSPOSE_H_

#include <algorithm>
#include <arm_neon.h>
#include <vector>

#include "deepvan/core/tensor.h"
#include "deepvan/export/deepvan.h"
#include "deepvan/utils/math.h"

namespace deepvan {
namespace transpose {

void TransposeNHWCToNCHWC3(const float *input,
                           float *output,
                           const index_t height,
                           const index_t width);

void TransposeNHWCToNCHWC3(const int *input,
                           int *output,
                           const index_t height,
                           const index_t width);

void TransposeNCHWToNHWCC2(const float *input,
                           float *output,
                           const index_t height,
                           const index_t width);

void TransposeNCHWToNHWCC2(const int *input,
                           int *output,
                           const index_t height,
                           const index_t width);

template <typename T>
void TransposeNDTensor(const T *X,
                       const std::vector<int64_t> &dims,
                       const std::vector<int> &axes,
                       T *Y) {
  int ndim = axes.size();
  std::vector<int64_t> Y_dims(ndim);
  for (int i = 0; i < ndim; ++i) {
    Y_dims[i] = dims[axes[i]];
  }
  // Measure amount of contiguous data we can copy at once
  int pivot = ndim - 1;
  int64_t block_size = 1;
  for (; pivot >= 0 && axes[pivot] == pivot; --pivot) {
    block_size *= Y_dims[pivot];
  }
  ++pivot;
  const int64_t num_blocks = std::accumulate(Y_dims.cbegin(),
                                             Y_dims.cbegin() + pivot,
                                             int64_t(1),
                                             std::multiplies<int64_t>());
  std::vector<int64_t> X_strides(pivot);
  ComputeTransposedStrides<int64_t>(
      pivot, dims.data(), axes.data(), X_strides.data());
  std::vector<int64_t> index(pivot, 0);
  // #pragma omp parallel for collapse(1) schedule(runtime)
  for (int64_t Y_index = 0; Y_index < num_blocks; ++Y_index) {
    const int64_t X_index = std::inner_product(
        X_strides.cbegin(), X_strides.cend(), index.cbegin(), int64_t(0));
    if (block_size == 1) {
      Y[Y_index] = X[X_index];
    } else {
      std::memcpy(Y + block_size * Y_index,
                  X + block_size * X_index,
                  block_size * sizeof(T));
    }
    IncreaseIndexInDims<int64_t>(pivot, Y_dims.data(), index.data());
  }
}

} // namespace transpose

template <typename T>
VanState Transpose(const T *input,
                   const std::vector<int64_t> &input_shape,
                   const std::vector<int> &dst_dims,
                   T *output,
                   DataType data_type = DataType::DT_FLOAT) {
  CONDITIONS(input_shape.size() == dst_dims.size(),
             "Only support the same ranks tensor transpose");

  std::vector<index_t> output_shape;
  for (size_t i = 0; i < dst_dims.size(); ++i) {
    output_shape.push_back(input_shape[dst_dims[i]]);
  }

  if (input_shape.size() == 2) {
    CONDITIONS(dst_dims[0] == 1 && dst_dims[1] == 0, "no need transform");
    index_t height = input_shape[0];
    index_t width = input_shape[1];
    index_t stride_i = height;
    index_t stride_j = width;
    index_t tile_size = height > 512 || width > 512 ? 64 : 32;
#pragma omp parallel for collapse(2)
    for (index_t i = 0; i < height; i += tile_size) {
      for (index_t j = 0; j < width; j += tile_size) {
        index_t end_i = std::min(i + tile_size, height);
        index_t end_j = std::min(j + tile_size, width);
        for (index_t tile_i = i; tile_i < end_i; ++tile_i) {
          for (index_t tile_j = j; tile_j < end_j; ++tile_j) {
            output[tile_j * stride_i + tile_i] =
                input[tile_i * stride_j + tile_j];
          }
        }
      }
    }
  } else if (input_shape.size() == 4) {
    std::vector<int> transpose_order_from_NHWC_to_NCHW{0, 3, 1, 2};
    std::vector<int> transpose_order_from_NCHW_to_NHWC{0, 2, 3, 1};
    index_t batch_size = input_shape[1] * input_shape[2] * input_shape[3];
    bool supported_dt =
        (data_type == DataType::DT_FLOAT || data_type == DataType::DT_INT32);

    if (dst_dims == transpose_order_from_NHWC_to_NCHW && input_shape[3] == 3 &&
        supported_dt) {
      for (index_t b = 0; b < input_shape[0]; ++b) {
        transpose::TransposeNHWCToNCHWC3(input + b * batch_size,
                                         output + b * batch_size,
                                         input_shape[1],
                                         input_shape[2]);
      }
    } else if (dst_dims == transpose_order_from_NHWC_to_NCHW && supported_dt) {
      auto height = input_shape[1];
      auto width = input_shape[2];
      auto channel = input_shape[3];
      index_t image_size = height * width;
#pragma omp parallel for collapse(2)
      for (index_t b = 0; b < input_shape[0]; ++b) {
        for (index_t h = 0; h < height; ++h) {
          for (index_t c = 0; c < channel; ++c) {
            index_t in_offset = h * width * channel;
            index_t out_offset = c * image_size + h * width;
            for (index_t w = 0; w < width; ++w) {
              output[out_offset + w] = input[in_offset + w * channel + c];
            }
          }
        }
      }
    } else if (dst_dims == transpose_order_from_NCHW_to_NHWC &&
               input_shape[1] == 2 && supported_dt) {
      for (index_t b = 0; b < input_shape[0]; ++b) {
        transpose::TransposeNCHWToNHWCC2(input + b * batch_size,
                                         output + b * batch_size,
                                         input_shape[2],
                                         input_shape[3]);
      }
    } else if (dst_dims == transpose_order_from_NCHW_to_NHWC && supported_dt) {
      const index_t channels = input_shape[1];
      const index_t height = input_shape[2];
      const index_t width = input_shape[3];
      const index_t image_size = height * width;
#pragma omp parallel for collapse(2)
      for (index_t b = 0; b < input_shape[0]; ++b) {
        for (index_t h = 0; h < height; ++h) {
          index_t in_offset = h * width;
          index_t out_offset = h * width * channels;
          for (index_t w = 0; w < width; ++w) {
            for (index_t c = 0; c < channels; ++c) {
              output[out_offset + w * channels + c] =
                  input[in_offset + c * image_size + w];

      // !opt for bert model
//       if (height * input_shape[0] < 16) {
//         CONDITIONS(input_shape[0] == 1 && channels % 8 == 0);
// #pragma omp parallel for schedule(runtime)
//         for (index_t c = 0; c < channels; c += 8) {
//           auto in_ptr0 =
//               reinterpret_cast<const float *>(input + c * image_size);
//           auto in_ptr1 =
//               reinterpret_cast<const float *>(input + (c + 1) * image_size);
//           auto in_ptr2 =
//               reinterpret_cast<const float *>(input + (c + 2) * image_size);
//           auto in_ptr3 =
//               reinterpret_cast<const float *>(input + (c + 3) * image_size);
//           auto in_ptr4 =
//               reinterpret_cast<const float *>(input + (c + 4) * image_size);
//           auto in_ptr5 =
//               reinterpret_cast<const float *>(input + (c + 5) * image_size);
//           auto in_ptr6 =
//               reinterpret_cast<const float *>(input + (c + 6) * image_size);
//           auto in_ptr7 =
//               reinterpret_cast<const float *>(input + (c + 7) * image_size);
//           for (index_t h = 0; h < height; ++h) {
//             index_t out_offset = h * width * channels;
//             auto out_ptr =
//                 reinterpret_cast<float *>(output + h * width * channels + c);
//             for (index_t w = 0; w < width; ++w) {
//               float32x4_t in = {*in_ptr0++, *in_ptr1++, *in_ptr2++, *in_ptr3++};
//               float32x4_t in1 = {
//                   *in_ptr4++, *in_ptr5++, *in_ptr6++, *in_ptr7++};
//               vst1q_f32(out_ptr, in);
//               vst1q_f32(out_ptr + 4, in1);
//               out_ptr += channels;
//             }
//           }
//         }
//       } else {
// #pragma omp parallel for collapse(2) schedule(runtime)
//         for (index_t b = 0; b < input_shape[0]; ++b) {
//           for (index_t h = 0; h < height; ++h) {
//             index_t in_offset = h * width;
//             index_t out_offset = h * width * channels;
//             for (index_t w = 0; w < width; ++w) {
//               for (index_t c = 0; c < channels; ++c) {
//                 output[out_offset + w * channels + c] =
//                     input[in_offset + c * image_size + w];
//              }
            }
          }
        }
      }
    } else if (dst_dims == std::vector<int>{0, 2, 1, 3}) {
      index_t height = input_shape[1];
      index_t width = input_shape[2];
      index_t channel = input_shape[3];
      index_t channel_raw_size = channel * sizeof(T);
      index_t stride_i = height;
      index_t stride_j = width;
      index_t tile_size =
          std::max(static_cast<index_t>(1),
                   static_cast<index_t>(std::sqrt(8 * 1024 / channel)));
#pragma omp parallel for collapse(2)
      for (index_t i = 0; i < height; i += tile_size) {
        for (index_t j = 0; j < width; j += tile_size) {
          index_t end_i = std::min(i + tile_size, height);
          index_t end_j = std::min(j + tile_size, width);
          for (index_t tile_i = i; tile_i < end_i; ++tile_i) {
            for (index_t tile_j = j; tile_j < end_j; ++tile_j) {
              memcpy(output + (tile_j * stride_i + tile_i) * channel,
                     input + (tile_i * stride_j + tile_j) * channel,
                     channel_raw_size);
//       if (height > 100) {
// #pragma omp parallel for schedule(runtime)
//         for (index_t i = 0; i < height; i += tile_size) {
//           for (index_t j = 0; j < width; j += tile_size) {
//             index_t end_i = std::min(i + tile_size, height);
//             index_t end_j = std::min(j + tile_size, width);
//             for (index_t tile_i = i; tile_i < end_i; ++tile_i) {
//               for (index_t tile_j = j; tile_j < end_j; ++tile_j) {
//                 memcpy(output + (tile_j * stride_i + tile_i) * channel,
//                        input + (tile_i * stride_j + tile_j) * channel,
//                        channel_raw_size);
//               }
//             }
//           }
//         }
//       } else {
// #pragma omp parallel for collapse(2) schedule(runtime)
//         for (index_t i = 0; i < height; i += tile_size) {
//           for (index_t j = 0; j < width; j += tile_size) {
//             index_t end_i = std::min(i + tile_size, height);
//             index_t end_j = std::min(j + tile_size, width);
//             for (index_t tile_i = i; tile_i < end_i; ++tile_i) {
//               for (index_t tile_j = j; tile_j < end_j; ++tile_j) {
//                 memcpy(output + (tile_j * stride_i + tile_i) * channel,
//                        input + (tile_i * stride_j + tile_j) * channel,
//                        channel_raw_size);
//               }
            }
          }
        }
      }
    } else {
      std::vector<index_t> in_stride{input_shape[1] * input_shape[2] *
                                         input_shape[3],
                                     input_shape[2] * input_shape[3],
                                     input_shape[3],
                                     1};
      std::vector<index_t> out_stride{output_shape[1] * output_shape[2] *
                                          output_shape[3],
                                      output_shape[2] * output_shape[3],
                                      output_shape[3],
                                      1};

      std::vector<index_t> idim(4, 0);
      std::vector<index_t> odim(4, 0);
      for (odim[0] = 0; odim[0] < output_shape[0]; ++odim[0]) {
        for (odim[1] = 0; odim[1] < output_shape[1]; ++odim[1]) {
          for (odim[2] = 0; odim[2] < output_shape[2]; ++odim[2]) {
            for (odim[3] = 0; odim[3] < output_shape[3]; ++odim[3]) {
              idim[dst_dims[0]] = odim[0];
              idim[dst_dims[1]] = odim[1];
              idim[dst_dims[2]] = odim[2];
              idim[dst_dims[3]] = odim[3];

              output[odim[0] * out_stride[0] + odim[1] * out_stride[1] +
                     odim[2] * out_stride[2] + odim[3]] =
                  input[idim[0] * in_stride[0] + idim[1] * in_stride[1] +
                        idim[2] * in_stride[2] + idim[3]];
            }
          }
        }
      }
    }
  } else if (input_shape.size() == 5) {
    std::vector<int> YOLO_RANK5{0, 1, 3, 4, 2};
    if (dst_dims == YOLO_RANK5) {
      const int dim0 = input_shape[0] * input_shape[1];
      const int dim1 = input_shape[2];
      const int dim2 = input_shape[3] * input_shape[4];
#pragma omp parallel for collapse(2)
      for (int ind0 = 0; ind0 < dim0; ind0++) {
        for (int ind1 = 0; ind1 < dim1; ind1++) {
          for (int ind2 = 0; ind2 < dim2; ind2++) {
            const int in_pos = (ind0 * dim1 + ind1) * dim2 + ind2;
            const int out_pos = (ind0 * dim2 + ind2) * dim1 + ind1;
            output[out_pos] = input[in_pos];
          } // ind2
        }   // ind1
      }     // ind0
    } else {
      transpose::TransposeNDTensor<T>(input, input_shape, dst_dims, output);
    }
  } else if (input_shape.size() == 6) {
    std::vector<int> RESO_RANK6{0, 1, 4, 2, 5, 3};
    if (dst_dims == RESO_RANK6) {
      const int dim0 = input_shape[0] * input_shape[1];
      const int dim1 = input_shape[2];
      const int dim2 = input_shape[3];
      const int dim3 = input_shape[4];
      const int dim4 = input_shape[5];
#pragma omp parallel for collapse(3)
      for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
          for (int k = 0; k < dim2; k++) {
            for (int l = 0; l < dim3; l++) {
              for (int m = 0; m < dim4; m++) {
                const int in_pos =
                    ((((i * dim1 + j) * dim2 + k) * dim3) + l) * dim4 + m;
                const int out_pos =
                    ((((i * dim3 + l) * dim1 + j) * dim4 + m) * dim2) + k;
                output[out_pos] = input[in_pos];
              } // m
            }   // l
          }     // k
        }       // j
      }         // i
    } else {
      UNSUPPORTED_OP("Transpose");
    }
  } else {
    transpose::TransposeNDTensor<T>(input, input_shape, dst_dims, output);
  }

  return VanState::SUCCEED;
}

} // namespace deepvan

#endif // DEEPVAN_BACKEND_COMMON_TRANSPOSE_H_
