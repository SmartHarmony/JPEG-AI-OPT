#ifdef NEON_SUPPORT
#endif // NEON_SUPPORT

#include "deepvan/backend/eltwise_op.h"
#include <algorithm>
#include <arm_neon.h>
#include <cmath>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "deepvan/backend/arm/common/tensor_logger.h"
#include "deepvan/backend/common/utils.h"
#include "deepvan/core/future.h"
#include "deepvan/core/operator.h"
#include "deepvan/core/tensor.h"
#include "deepvan/utils/memory.h"
// #include "deepvan/utils/quantize.h"
#ifdef OPENCL_SUPPORT
#include "deepvan/backend/opencl/buffer_transformer.h"
// #include "deepvan/backend/opencl/buffer_transformer_c3d.h"
#include "deepvan/backend/opencl/image/eltwise.h"
#endif // OPENCL_SUPPORT

// reference: https://lufficc.com/blog/tensorflow-and-numpy-broadcasting
// pow/exp: http://gruntthepeon.free.fr/ssemath/neon_mathfun.html

namespace deepvan {

inline index_t GetIndex(const std::vector<index_t> &shape,
                        const std::vector<index_t> &index) {
  index_t idx = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] > 1) {
      idx = idx * shape[i] + index[i];
    }
  }
  return idx;
}

inline void IncreaseIndex(const std::vector<index_t> &shape,
                          std::vector<index_t> *index) {
  for (index_t i = static_cast<index_t>(shape.size()) - 1; i >= 0; --i) {
    ++(*index)[i];
    if ((*index)[i] >= shape[i]) {
      (*index)[i] -= shape[i];
    } else {
      break;
    }
  }
}

template <typename T, typename DstType>
inline void
TensorGeneralBroadcastEltwise(const EltwiseType type,
                              const T *input0,
                              const T *input1,
                              const std::vector<float> &coeff,
                              const bool swapped,
                              const std::vector<index_t> &input0_shape,
                              const std::vector<index_t> &input1_shape,
                              const std::vector<index_t> &output_shape,
                              DstType *output) {
  const index_t output_size = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<index_t>());
  std::vector<index_t> out_index(output_shape.size(), 0);
  switch (type) {
  case SUM:
    if (coeff.empty()) {
      for (index_t i = 0; i < output_size; ++i) {
        const index_t idx0 = GetIndex(input0_shape, out_index);
        const index_t idx1 = GetIndex(input1_shape, out_index);
        output[i] = input0[idx0] + input1[idx1];
        IncreaseIndex(output_shape, &out_index);
      }
    } else {
      std::vector<float> coeff_copy = coeff;
      if (swapped) {
        std::swap(coeff_copy[0], coeff_copy[1]);
      }
      for (index_t i = 0; i < output_size; ++i) {
        const index_t idx0 = GetIndex(input0_shape, out_index);
        const index_t idx1 = GetIndex(input1_shape, out_index);
        output[i] = input0[idx0] * coeff_copy[0] + input1[idx1] * coeff_copy[1];
        IncreaseIndex(output_shape, &out_index);
      }
    }
    break;
  case SUB:
    if (!swapped) {
      for (index_t i = 0; i < output_size; ++i) {
        const index_t idx0 = GetIndex(input0_shape, out_index);
        const index_t idx1 = GetIndex(input1_shape, out_index);
        output[i] = input0[idx0] - input1[idx1];
        IncreaseIndex(output_shape, &out_index);
      }
    } else {
      for (index_t i = 0; i < output_size; ++i) {
        const index_t idx0 = GetIndex(input0_shape, out_index);
        const index_t idx1 = GetIndex(input1_shape, out_index);
        output[i] = input1[idx1] - input0[idx0];
        IncreaseIndex(output_shape, &out_index);
      }
    }
    break;
  case PROD:
    for (index_t i = 0; i < output_size; ++i) {
      const index_t idx0 = GetIndex(input0_shape, out_index);
      const index_t idx1 = GetIndex(input1_shape, out_index);
      output[i] = input0[idx0] * input1[idx1];
      IncreaseIndex(output_shape, &out_index);
    }
    break;
  case DIV:
    if (!swapped) {
      for (index_t i = 0; i < output_size; ++i) {
        const index_t idx0 = GetIndex(input0_shape, out_index);
        const index_t idx1 = GetIndex(input1_shape, out_index);
        output[i] = input0[idx0] / input1[idx1];
        IncreaseIndex(output_shape, &out_index);
      }
    } else {
      for (index_t i = 0; i < output_size; ++i) {
        const index_t idx0 = GetIndex(input0_shape, out_index);
        const index_t idx1 = GetIndex(input1_shape, out_index);
        output[i] = input1[idx1] / input0[idx0];
        IncreaseIndex(output_shape, &out_index);
      }
    }
    break;
  case FLOOR_DIV:
    if (!swapped) {
      for (index_t i = 0; i < output_size; ++i) {
        const index_t idx0 = GetIndex(input0_shape, out_index);
        const index_t idx1 = GetIndex(input1_shape, out_index);
        output[i] = std::floor(input0[idx0] / input1[idx1]);
        IncreaseIndex(output_shape, &out_index);
      }
    } else {
      for (index_t i = 0; i < output_size; ++i) {
        const index_t idx0 = GetIndex(input0_shape, out_index);
        const index_t idx1 = GetIndex(input1_shape, out_index);
        output[i] = std::floor(input1[idx1] / input0[idx0]);
        IncreaseIndex(output_shape, &out_index);
      }
    }
    break;
  case MIN:
    for (index_t i = 0; i < output_size; ++i) {
      const index_t idx0 = GetIndex(input0_shape, out_index);
      const index_t idx1 = GetIndex(input1_shape, out_index);
      output[i] = std::min(input1[idx1], input0[idx0]);
      IncreaseIndex(output_shape, &out_index);
    }
    break;
  case MAX:
    for (index_t i = 0; i < output_size; ++i) {
      const index_t idx0 = GetIndex(input0_shape, out_index);
      const index_t idx1 = GetIndex(input1_shape, out_index);
      output[i] = std::max(input1[idx1], input0[idx0]);
      IncreaseIndex(output_shape, &out_index);
    }
    break;
  case SQR_DIFF:
    for (index_t i = 0; i < output_size; ++i) {
      const index_t idx0 = GetIndex(input0_shape, out_index);
      const index_t idx1 = GetIndex(input1_shape, out_index);
      output[i] = std::pow(input1[idx1] - input0[idx0], 2.f);
      IncreaseIndex(output_shape, &out_index);
    }
    break;
  case POW:
    if (!swapped) {
      for (index_t i = 0; i < output_size; ++i) {
        const index_t idx0 = GetIndex(input0_shape, out_index);
        const index_t idx1 = GetIndex(input1_shape, out_index);
        output[i] = std::pow(input0[idx0], input1[idx1]);
        IncreaseIndex(output_shape, &out_index);
      }
    } else {
      for (index_t i = 0; i < output_size; ++i) {
        const index_t idx0 = GetIndex(input0_shape, out_index);
        const index_t idx1 = GetIndex(input1_shape, out_index);
        output[i] = std::pow(input1[idx1], input0[idx0]);
        IncreaseIndex(output_shape, &out_index);
      }
    }
    break;
  case EQUAL:
    for (index_t i = 0; i < output_size; ++i) {
      const index_t idx0 = GetIndex(input0_shape, out_index);
      const index_t idx1 = GetIndex(input1_shape, out_index);
      output[i] = input1[idx1] == input0[idx0];
      IncreaseIndex(output_shape, &out_index);
    }
    break;
  case EXP: CONDITIONS(false, "Exp cannot broadcast"); break;
  case ERF: CONDITIONS(false, "ERF cannot broadcast"); break;
  default: LOG(FATAL) << "Eltwise op not support type " << type;
  }
}

template <typename T, typename DstType>
inline void TensorBroadcastEltwise(const EltwiseType type,
                                   const T *input0,
                                   const T *input1,
                                   const std::vector<float> &coeff,
                                   const index_t diff_size,
                                   const index_t common_size,
                                   const bool swapped,
                                   DstType *output) {
  switch (type) {
  case SUM:
    if (coeff.empty()) {
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t d = 0; d < diff_size; ++d) {
        for (index_t i = 0; i < common_size; ++i) {
          output[i + d * common_size] = input0[i + d * common_size] + input1[i];
        }
      }
    } else {
      std::vector<float> coeff_copy = coeff;
      if (swapped) {
        std::swap(coeff_copy[0], coeff_copy[1]);
      }
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t d = 0; d < diff_size; ++d) {
        for (index_t i = 0; i < common_size; ++i) {
          output[i + d * common_size] =
              input0[i + d * common_size] * coeff_copy[0] +
              input1[i] * coeff_copy[1];
        }
      }
    }
    break;
  case SUB:
    if (!swapped) {
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t d = 0; d < diff_size; ++d) {
        for (index_t i = 0; i < common_size; ++i) {
          output[i + d * common_size] = input0[i + d * common_size] - input1[i];
        }
      }
    } else {
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t d = 0; d < diff_size; ++d) {
        for (index_t i = 0; i < common_size; ++i) {
          output[i + d * common_size] = input1[i] - input0[i + d * common_size];
        }
      }
    }
    break;
  case PROD:
#pragma omp parallel for collapse(2) schedule(runtime)
    for (index_t d = 0; d < diff_size; ++d) {
      for (index_t i = 0; i < common_size; ++i) {
        output[i + d * common_size] = input0[i + d * common_size] * input1[i];
      }
    }
    // bert code
    // #pragma omp parallel for collapse(1) schedule(runtime)
    //     for (index_t d = 0; d < diff_size; ++d) {
    //       index_t pos = d * common_size;
    //       for (index_t i = 0; i < common_size; ++i) {
    //         output[pos] = input0[pos] * input1[i];
    //         ++pos;
    //       }
    //     }
    break;
  case DIV:
    if (!swapped) {
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t d = 0; d < diff_size; ++d) {
        for (index_t i = 0; i < common_size; ++i) {
          output[i + d * common_size] = input0[i + d * common_size] / input1[i];
        }
      }
    } else {
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t d = 0; d < diff_size; ++d) {
        for (index_t i = 0; i < common_size; ++i) {
          output[i + d * common_size] = input1[i] / input0[i + d * common_size];
        }
      }
    }
    break;
  case FLOOR_DIV:
    if (!swapped) {
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t d = 0; d < diff_size; ++d) {
        for (index_t i = 0; i < common_size; ++i) {
          output[i + d * common_size] =
              std::floor(input0[i + d * common_size] / input1[i]);
        }
      }
    } else {
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t d = 0; d < diff_size; ++d) {
        for (index_t i = 0; i < common_size; ++i) {
          output[i + d * common_size] =
              std::floor(input1[i] / input0[i + d * common_size]);
        }
      }
    }
    break;
  case MIN:
#pragma omp parallel for collapse(2) schedule(runtime)
    for (index_t d = 0; d < diff_size; ++d) {
      for (index_t i = 0; i < common_size; ++i) {
        output[i + d * common_size] =
            std::min(input0[i + d * common_size], input1[i]);
      }
    }
    break;
  case MAX:
#pragma omp parallel for collapse(2) schedule(runtime)
    for (index_t d = 0; d < diff_size; ++d) {
      for (index_t i = 0; i < common_size; ++i) {
        output[i + d * common_size] =
            std::max(input0[i + d * common_size], input1[i]);
      }
    }
    break;
  case SQR_DIFF:
#pragma omp parallel for collapse(2) schedule(runtime)
    for (index_t d = 0; d < diff_size; ++d) {
      for (index_t i = 0; i < common_size; ++i) {
        output[i + d * common_size] =
            std::pow(input0[i + d * common_size] - input1[i], 2.f);
      }
    }
    break;
  case POW:
    if (!swapped) {
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t d = 0; d < diff_size; ++d) {
        for (index_t i = 0; i < common_size; ++i) {
          output[i + d * common_size] =
              std::pow(input0[i + d * common_size], input1[i]);
        }
      }
    } else {
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t d = 0; d < diff_size; ++d) {
        for (index_t i = 0; i < common_size; ++i) {
          output[i + d * common_size] =
              std::pow(input1[i], input0[i + d * common_size]);
        }
      }
    }
    break;
  case NEG:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < diff_size * common_size; ++i) {
      output[i] = -input0[i];
    }
    break;
  case ABS:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < diff_size * common_size; ++i) {
      output[i] = std::fabs(input0[i]);
    }
    break;
  case EQUAL:
#pragma omp parallel for collapse(2) schedule(runtime)
    for (index_t d = 0; d < diff_size; ++d) {
      for (index_t i = 0; i < common_size; ++i) {
        output[i + d * common_size] = input0[i + d * common_size] == input1[i];
      }
    }
    break;
  case EXP: CONDITIONS(false, "Exp cannot broadcast"); break;
  case ERF: CONDITIONS(false, "ERF cannot broadcast"); break;
  default: LOG(FATAL) << "Eltwise op not support type " << type;
  }
}

// Multiplication is costly, so we specialize the following case.
template <typename T, typename DstType>
inline void TensorEltwise(const EltwiseType type,
                          const T *input0,
                          const T *input1,
                          const std::vector<float> &coeff,
                          const index_t size,
                          const bool swapped,
                          DstType *output) {
  switch (type) {
  case SUM:
    if (coeff.empty()) {
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] + input1[i];
      }

    } else {
      std::vector<float> coeff_copy = coeff;
      if (swapped) {
        std::swap(coeff_copy[0], coeff_copy[1]);
      }
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] * coeff_copy[0] + input1[i] * coeff_copy[1];
      }
    }
    break;
  case SUB:
    if (!swapped) {
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] - input1[i];
      }

    } else {
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = input1[i] - input0[i];
      }
    }
    break;
  case PROD:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output[i] = input0[i] * input1[i];
    }

    break;
  case DIV:
    if (!swapped) {
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] / input1[i];
      }

    } else {
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = input1[i] / input0[i];
      }
    }
    break;
  case FLOOR_DIV:
    if (!swapped) {
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::floor(input0[i] / input1[i]);
      }
    } else {
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::floor(input1[i] / input0[i]);
      }
    }
    break;
  case MIN:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output[i] = std::min(input0[i], input1[i]);
    }

    break;
  case MAX:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output[i] = std::max(input0[i], input1[i]);
    }

    break;
  case SQR_DIFF:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output[i] = std::pow(input0[i] - input1[i], 2.f);
    }

    break;
  case POW:
    if (!swapped) {
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::pow(input0[i], input1[i]);
      }
    } else {
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::pow(input1[i], input0[i]);
      }
    }
    break;
  case NEG:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output[i] = -input0[i];
    }
    break;
  case ABS:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output[i] = std::fabs(input0[i]);
    }
    break;
  case EQUAL:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output[i] = input0[i] == input1[i];
    }
    break;
  case EXP: CONDITIONS(false, "Exp only has scalar operation"); break;
  case ERF: CONDITIONS(false, "ERF only has scalar operation"); break;
  default: LOG(FATAL) << "Eltwise op not support type " << type;
  }
}

// Multiplication is costly, so we specialize the following case.
template <typename T, typename DstType>
inline void TensorScalarEltwise(const EltwiseType type,
                                const T *input0,
                                const T input1,
                                const std::vector<float> &coeff,
                                const index_t size,
                                const bool swapped,
                                DstType *output) {
  // index_t batch = 64;
  // index_t count = size / batch;
  switch (type) {
  case SUM:
    if (coeff.empty()) {
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] + input1;
      }

    } else {
      std::vector<float> coeff_copy = coeff;
      if (swapped) {
        std::swap(coeff_copy[0], coeff_copy[1]);
      }
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] * coeff_copy[0] + input1 * coeff_copy[1];
      }
    }
    break;
  case SUB:
    if (!swapped) {
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] - input1;
      }

    } else {
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = input1 - input0[i];
      }
    }
    break;
  case PROD: // CONDITIONS(size % batch == 0, "Error");
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output[i] = input0[i] * input1;
    }

    // for bert
    //  #pragma omp parallel for schedule(runtime)
    //      for (index_t b = 0; b < batch; ++b) {
    //        index_t start = b * count;
    //        index_t end = (b + 1) * count;
    //        for (index_t i = start; i < end; ++i) {
    //          output[i] = input0[i] * input1;
    //        }
    //      }

    break;
  case DIV:
    if (!swapped) {
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = input0[i] / input1;
      }

    } else {
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = input1 / input0[i];
      }
    }
    break;
  case FLOOR_DIV:
    if (!swapped) {
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::floor(input0[i] / input1);
      }
    } else {
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::floor(input1 / input0[i]);
      }
    }
    break;
  case MIN:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output[i] = std::min(input0[i], input1);
    }

    break;
  case MAX:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output[i] = std::max(input0[i], input1);
    }

    break;
  case SQR_DIFF:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output[i] = std::pow(input0[i] - input1, 2.f);
    }

    break;
  case POW:
    if (!swapped) {
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::pow(input0[i], input1);
      }
    } else {
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output[i] = std::pow(input1, input0[i]);
      }
    }
    break;
  case NEG:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output[i] = -input0[i];
    }
    break;
  case ABS:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output[i] = std::fabs(input0[i]);
    }
    break;
  case EQUAL:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output[i] = input0[i] == input1;
    }

    break;
  case EXP:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output[i] = std::exp(input0[i]);
    }
    break;
  case ERF:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output[i] = std::erf(input0[i]);
    }
    break;
  default: LOG(FATAL) << "Eltwise op not support type " << type;
  }
}

template <typename T, typename DstType>
inline void TensorEltwisePerChannel(const EltwiseType type,
                                    const T *input0,
                                    const T *input1,
                                    const std::vector<float> &coeff,
                                    const index_t batch0,
                                    const index_t batch1,
                                    const index_t channel,
                                    const index_t image_size,
                                    const bool swapped,
                                    DstType *output) {
  switch (type) {
  case SUM:
    if (coeff.empty()) {
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t b = 0; b < batch0; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
          const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
          DstType *out_ptr = output + ((b * channel) + c) * image_size;
          for (index_t i = 0; i < image_size; ++i) {
            out_ptr[i] = in0_ptr[i] + in1_ptr[c];
          }
        }
      }
    } else {
      std::vector<float> coeff_copy = coeff;
      if (swapped) {
        std::swap(coeff_copy[0], coeff_copy[1]);
      }
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t b = 0; b < batch0; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
          const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
          DstType *out_ptr = output + ((b * channel) + c) * image_size;
          for (index_t i = 0; i < image_size; ++i) {
            out_ptr[i] =
                in0_ptr[i] * coeff_copy[0] + in1_ptr[c] * coeff_copy[1];
          }
        }
      }
    }
    break;
  case SUB:
    if (!swapped) {
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t b = 0; b < batch0; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
          const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
          DstType *out_ptr = output + ((b * channel) + c) * image_size;
          for (index_t i = 0; i < image_size; ++i) {
            out_ptr[i] = in0_ptr[i] - in1_ptr[c];
          }
        }
      }
    } else {
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t b = 0; b < batch0; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
          const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
          DstType *out_ptr = output + ((b * channel) + c) * image_size;
          for (index_t i = 0; i < image_size; ++i) {
            out_ptr[i] = in1_ptr[c] - in0_ptr[i];
          }
        }
      }
    }
    break;
  case PROD:
#pragma omp parallel for collapse(2) schedule(runtime)
    for (index_t b = 0; b < batch0; ++b) {
      for (index_t c = 0; c < channel; ++c) {
        const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
        const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
        DstType *out_ptr = output + ((b * channel) + c) * image_size;
        for (index_t i = 0; i < image_size; ++i) {
          out_ptr[i] = in0_ptr[i] * in1_ptr[c];
        }
      }
    }
    break;
  case DIV:
    if (!swapped) {
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t b = 0; b < batch0; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
          const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
          DstType *out_ptr = output + ((b * channel) + c) * image_size;
          for (index_t i = 0; i < image_size; ++i) {
            out_ptr[i] = in0_ptr[i] / in1_ptr[c];
          }
        }
      }
    } else {
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t b = 0; b < batch0; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
          const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
          DstType *out_ptr = output + ((b * channel) + c) * image_size;
          for (index_t i = 0; i < image_size; ++i) {
            out_ptr[i] = in1_ptr[c] / in0_ptr[i];
          }
        }
      }
    }
    break;
  case FLOOR_DIV:
    if (!swapped) {
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t b = 0; b < batch0; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
          const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
          DstType *out_ptr = output + ((b * channel) + c) * image_size;
          for (index_t i = 0; i < image_size; ++i) {
            out_ptr[i] = std::floor(in0_ptr[i] / in1_ptr[c]);
          }
        }
      }
    } else {
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t b = 0; b < batch0; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
          const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
          DstType *out_ptr = output + ((b * channel) + c) * image_size;
          for (index_t i = 0; i < image_size; ++i) {
            out_ptr[i] = std::floor(in1_ptr[c] / in0_ptr[i]);
          }
        }
      }
    }
    break;
  case MIN:
#pragma omp parallel for collapse(2) schedule(runtime)
    for (index_t b = 0; b < batch0; ++b) {
      for (index_t c = 0; c < channel; ++c) {
        const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
        const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
        DstType *out_ptr = output + ((b * channel) + c) * image_size;
        for (index_t i = 0; i < image_size; ++i) {
          out_ptr[i] = std::min(in0_ptr[i], in1_ptr[c]);
        }
      }
    }
    break;
  case MAX:
#pragma omp parallel for collapse(2) schedule(runtime)
    for (index_t b = 0; b < batch0; ++b) {
      for (index_t c = 0; c < channel; ++c) {
        const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
        const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
        DstType *out_ptr = output + ((b * channel) + c) * image_size;
        for (index_t i = 0; i < image_size; ++i) {
          out_ptr[i] = std::max(in0_ptr[i], in1_ptr[c]);
        }
      }
    }
    break;
  case SQR_DIFF:
#pragma omp parallel for collapse(2) schedule(runtime)
    for (index_t b = 0; b < batch0; ++b) {
      for (index_t c = 0; c < channel; ++c) {
        const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
        const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
        DstType *out_ptr = output + ((b * channel) + c) * image_size;
        for (index_t i = 0; i < image_size; ++i) {
          out_ptr[i] = std::pow(in0_ptr[i] - in1_ptr[c], 2.f);
        }
      }
    }
    break;
  case POW:
    if (!swapped) {
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t b = 0; b < batch0; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
          const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
          DstType *out_ptr = output + ((b * channel) + c) * image_size;
          for (index_t i = 0; i < image_size; ++i) {
            out_ptr[i] = std::pow(in0_ptr[i], in1_ptr[c]);
          }
        }
      }
    } else {
#pragma omp parallel for collapse(2) schedule(runtime)
      for (index_t b = 0; b < batch0; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
          const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
          DstType *out_ptr = output + ((b * channel) + c) * image_size;
          for (index_t i = 0; i < image_size; ++i) {
            out_ptr[i] = std::pow(in1_ptr[c], in0_ptr[i]);
          }
        }
      }
    }
    break;
  case NEG:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < batch0 * channel * image_size; ++i) {
      output[i] = -input0[i];
    }
    break;
  case ABS:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < batch0 * channel * image_size; ++i) {
      output[i] = std::fabs(input0[i]);
    }
    break;
  case EQUAL:
#pragma omp parallel for collapse(2) schedule(runtime)
    for (index_t b = 0; b < batch0; ++b) {
      for (index_t c = 0; c < channel; ++c) {
        const T *in0_ptr = input0 + ((b * channel) + c) * image_size;
        const T *in1_ptr = input1 + (batch1 > 1 ? b * channel : 0);
        DstType *out_ptr = output + ((b * channel) + c) * image_size;
        for (index_t i = 0; i < image_size; ++i) {
          out_ptr[i] = in0_ptr[i] == in1_ptr[c];
        }
      }
    }
    break;
  case EXP:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < batch0 * channel * image_size; ++i) {
      output[i] = std::exp(input0[i]);
    }
    break;
  case ERF:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < batch0 * channel * image_size; ++i) {
      output[i] = std::erf(input0[i]);
    }
    break;
  default: LOG(FATAL) << "Eltwise op not support type " << type;
  }
}

template <DeviceType D, class T>
class EltwiseOp : public Operation {
public:
  explicit EltwiseOp(OpConstructContext *context)
      : Operation(context),
        type_(static_cast<EltwiseType>(Operation::GetOptionalArg<int>(
            "type",
            static_cast<int>(EltwiseType::NONE)))),
        coeff_(Operation::GetRepeatedArgs<float>("coeff")),
        scalar_input_(Operation::GetOptionalArg<float>("scalar_input", 1.0)),
        scalar_input_index_(
            Operation::GetOptionalArg<int32_t>("scalar_input_index", 1)),
        has_data_format_(Operation::GetOptionalArg<int>("has_data_format", 0)) {
  }

  VanState Run(OpContext *context) override {
    UNUSED_VARIABLE(context);
    const Tensor *input0 = this->Input(0);
    const Tensor *input1 = this->InputSize() == 2 ? this->Input(1) : nullptr;
    common::utils::GetFallbackTensor(input1, context, true);
    Tensor *output = this->Output(0);
    if (input1 == nullptr) {
      scalar_tensor_.Resize({});
      Tensor::MappingGuard guard(&scalar_tensor_);
      auto scalar_data = scalar_tensor_.mutable_data<T>();
      scalar_data[0] = static_cast<T>(scalar_input_);
      input1 = &scalar_tensor_;
    }

    if (IsLogicalType(type_)) {
      // as we do not have bool-type tensor, we use int type
      return DoEltwise<int32_t>(input0, input1, output);
    } else {
      return DoEltwise<T>(input0, input1, output);
    }
  }

private:
  template <typename DstType>
  VanState DoEltwise(const Tensor *input0,
                     const Tensor *input1,
                     Tensor *output) {
    bool swapped = false;
    if (input0->dim_size() < input1->dim_size() ||
        (input0->dim_size() == input1->dim_size() &&
         input0->size() < input1->size())) {
      std::swap(input0, input1);
      swapped = true;
    }
    if (scalar_input_index_ == 0) {
      swapped = !swapped;
    }

    // check if we can broadcast tensor
    uint32_t rank_diff =
        static_cast<uint32_t>(input0->dim_size() - input1->dim_size());
    has_data_format_ = input1->size() == 1 ? false : has_data_format_;
    if (has_data_format_) {
      bool cond = false;
      for (int i : {4, 5}) {
        cond |= input0->dim_size() == i &&
                (input1->dim_size() == 0 ||
                 (input1->dim_size() == i && input1->dim(1) == input0->dim(1) &&
                  (input1->dim(0) == input0->dim(0) || input1->dim(0) == 1)) ||
                 (input1->dim_size() == 1 && input1->dim(0) == input0->dim(1)));
      }

      auto in0_ptr = input0->shape().end();
      auto in1_ptr = input1->shape().end();
      if (input0->shape().size() <= 3 && input1->shape().size()) {
        for (int i = fmin(input0->shape().size(), input1->shape().size());
             i > 0;
             i--) {
          in0_ptr--;
          in1_ptr--;
          if (*in0_ptr == *in1_ptr || *in0_ptr == 1 || *in1_ptr == 1) {
            cond = true;
            break;
          }
        }
      }

      CONDITIONS(cond, "only support broadcast channel dimension ")
          << operator_name() << " shape: " << MakeString(input0->shape())
          << " VS " << MakeString(input1->shape());

    } else {
      for (uint32_t i = 0; i < input1->dim_size(); ++i) {
        CONDITIONS(input0->dim(rank_diff + i) == 1 || input1->dim(i) == 1 ||
                       input0->dim(rank_diff + i) == input1->dim(i),
                   "Element-Wise op only support tail dimensions broadcast");
      }
    }

    // !checkme !important: comment this when benchmark bertsquad8
    Tensor::MappingGuard input0_guard(input0);
    Tensor::MappingGuard input1_guard(input1);

    const T *input0_ptr = input0->data<T>();
    const T *input1_ptr = input1->data<T>();

    if (has_data_format_ && input1->dim_size() > 0) {
      RETURN_IF_ERROR(output->ResizeLike(input0));
      Tensor::MappingGuard output_guard(output);
      DstType *output_ptr = output->mutable_data<DstType>();
      if (input1->size() < input0->size()) {
        auto in0_shape = input0->shape();
        TensorEltwisePerChannel(type_,
                                input0_ptr,
                                input1_ptr,
                                coeff_,
                                input0->dim(0),
                                input1->dim_size() == 1 ? 1 : input1->dim(0),
                                input0->dim(1),
                                std::accumulate(in0_shape.begin() + 2,
                                                in0_shape.end(),
                                                1,
                                                std::multiplies<index_t>()),
                                swapped,
                                output_ptr);
      } else {
        TensorEltwise(type_,
                      input0_ptr,
                      input1_ptr,
                      coeff_,
                      input0->size(),
                      swapped,
                      output_ptr);
      }
    } else {
      const std::vector<index_t> &input0_shape = input0->shape();
      std::vector<index_t> input1_shape(rank_diff, 1);
      input1_shape.insert(
          input1_shape.end(), input1->shape().begin(), input1->shape().end());

      std::vector<index_t> output_shape(input0->dim_size(), 0);
      for (unsigned int i = 0; i < input0_shape.size(); ++i) {
        output_shape[i] = std::max(input0_shape[i], input1_shape[i]);
      }
      RETURN_IF_ERROR(output->Resize(output_shape));
      Tensor::MappingGuard output_guard(output);
      DstType *output_ptr = output->mutable_data<DstType>();

      bool need_general_broadcast = false;
      for (uint32_t i = 0; i < input1->dim_size(); ++i) {
        if ((input0->dim(rank_diff + i) == 1 && input1->dim(i) > 1) ||
            (input0->dim(rank_diff + i) > 1 && input1->dim(i) == 1)) {
          need_general_broadcast = true;
          break;
        }
      }

      // CHECKME - EXP
      if (input1->size() == 1) {
        VLOG(INFO) << operator_name() << ", type: " << type_;
        TensorScalarEltwise(type_,
                            input0_ptr,
                            input1_ptr[0],
                            coeff_,
                            input0->size(),
                            swapped,
                            output_ptr);
      } else if (input0_shape == input1_shape) {
        TensorEltwise(type_,
                      input0_ptr,
                      input1_ptr,
                      coeff_,
                      input0->size(),
                      swapped,
                      output_ptr);
        // } else if (need_general_broadcast || input1->size() < input0->size())
        // {
      } else if (need_general_broadcast) {
        // !tricky for yolo
        if (input0_shape.size() == 2 && input1_shape.size() == 2 &&
            input1_shape[1] == 1 && input1_shape[0] == input0_shape[0] &&
            type_ == EltwiseType::PROD) {
#pragma omp parallel for schedule(runtime)
          for (int i = 0; i < output_shape[0]; i++) {
            T value = input1_ptr[i];
            const T *input0_data = input0_ptr + i * input0_shape[1];
            DstType *output_data = output_ptr + i * output_shape[1];
            for (int j = 0; j < output_shape[1]; j++) {
              output_data[j] = input0_data[j] * value;
            }
          }
        } else if (std::accumulate(input1_shape.cbegin(),
                                   input1_shape.cend(),
                                   1,
                                   std::multiplies<index_t>()) ==
                   *input1_shape.rbegin()) { // !tricky for mobilebert
          const index_t fused_batch =
              std::accumulate(output_shape.begin(),
                              output_shape.end() - 1,
                              1,
                              std::multiplies<index_t>());
          const index_t channels = *output_shape.rbegin();
          switch (type_) {
          case SUM:
#pragma omp parallel for schedule(runtime)
            for (int n = 0; n < fused_batch; n++) {
              index_t pos = n * channels;
              for (int c = 0; c < channels; c++, pos++) {
                output_ptr[pos] = input0_ptr[pos] + input1_ptr[c];
              }
            }
            break;
          case PROD:
#pragma omp parallel for schedule(runtime)
            for (int n = 0; n < fused_batch; n++) {
              index_t pos = n * channels;
              for (int c = 0; c < channels; c++, pos++) {
                output_ptr[pos] = input0_ptr[pos] * input1_ptr[c];
              }
            }
            break;
          default: STUB;
          }

        } else if (input0_shape.size() == 3 && input1_shape.size() == 3 &&
                   std::accumulate(input1_shape.cbegin(),
                                   input1_shape.cend(),
                                   1,
                                   std::multiplies<index_t>()) ==
                       *(input1_shape.end() - 2)) { // !tricky for distilbert
          const index_t fused_batch =
              std::accumulate(output_shape.begin(),
                              output_shape.end() - 1,
                              1,
                              std::multiplies<index_t>());
          const index_t channels = *output_shape.rbegin();
          switch (type_) {
          case SUB:
#pragma omp parallel for schedule(runtime)
            for (int n = 0; n < fused_batch; n++) {
              index_t pos = n * channels;
              T value = input1_ptr[n];
              for (int c = 0; c < channels; c++, pos++) {
                output_ptr[pos] = input0_ptr[pos] - value;
              }
            }
            break;
          case DIV:
#pragma omp parallel for schedule(runtime)
            for (int n = 0; n < fused_batch; n++) {
              index_t pos = n * channels;
              T value = input1_ptr[n];
              for (int c = 0; c < channels; c++, pos++) {
                output_ptr[pos] = input0_ptr[pos] / value;
              }
            }
            break;
          case PROD:
#pragma omp parallel for schedule(runtime)
            for (int n = 0; n < fused_batch; n++) {
              index_t pos = n * channels;
              T value = input1_ptr[n];
              for (int c = 0; c < channels; c++, pos++) {
                output_ptr[pos] = input0_ptr[pos] * value;
              }
            }
            break;
          default:
            LOG(INFO) << "Eltwise: " << operator_name() << type_;
            STUB;
            break;
          }
        } else {
          TensorGeneralBroadcastEltwise(type_,
                                        input0_ptr,
                                        input1_ptr,
                                        coeff_,
                                        swapped,
                                        input0_shape,
                                        input1_shape,
                                        output_shape,
                                        output_ptr);
        }
      } else {
        index_t common_size = input1->size();
        index_t diff_size = input0->size() / common_size;
        TensorBroadcastEltwise(type_,
                               input0_ptr,
                               input1_ptr,
                               coeff_,
                               diff_size,
                               common_size,
                               swapped,
                               output_ptr);
      }
    }
    // debugger::WriteTensor2File(output, output->name());
    return VanState::SUCCEED;
  }

private:
  EltwiseType type_;
  std::vector<float> coeff_;
  float scalar_input_;
  int32_t scalar_input_index_;
  int has_data_format_;
  Tensor scalar_tensor_;
};

#ifdef OPENCL_SUPPORT
template <typename T>
class EltwiseOp<DeviceType::GPU, T> : public Operation {
public:
  explicit EltwiseOp(OpConstructContext *context) : Operation(context) {
    EltwiseType type = static_cast<EltwiseType>(Operation::GetOptionalArg<int>(
        "type", static_cast<int>(EltwiseType::NONE)));
    NetworkController *ws = context->workspace();
    std::vector<float> coeff = Operation::GetRepeatedArgs<float>("coeff");
    float scalar_input = Operation::GetOptionalArg<float>("scalar_input", 1.0);
    int32_t scalar_input_index =
        Operation::GetOptionalArg<int32_t>("scalar_input_index", 1);
    MemoryType mem_type;
    if (context->device()->gpu_runtime()->UseImageMemory()) {
      mem_type = MemoryType::GPU_IMAGE;
      // C3D NOT SAFE CHECKME @vgod

      kernel_ = make_unique<opencl::image::EltwiseKernel<T>>(
          type, coeff, scalar_input, scalar_input_index, operator_name());
    }

    context->set_output_mem_type(mem_type);
    // Transform filters
    int input_size = operator_def_->input_size();
    for (int i = 0; i < input_size; ++i) {
      if (ws->HasTensor(operator_def_->input(i)) &&
          ws->GetTensor(operator_def_->input(i))->is_weight()) {
        if (ws->GetTensor(operator_def_->input(i))->dim_size() == 1) {
          CONDITIONS(TransformFilter<T>(context,
                                        operator_def_.get(),
                                        i,
                                        OpenCLBufferType::ARGUMENT,
                                        mem_type,
                                        0,
                                        pruning_type_) == VanState::SUCCEED);
        } else if (ws->GetTensor(operator_def_->input(i))->dim_size() == 4) {
          CONDITIONS(TransformFilter<T>(context,
                                        operator_def_.get(),
                                        i,
                                        OpenCLBufferType::IN_OUT_CHANNEL,
                                        mem_type,
                                        0,
                                        pruning_type_) == VanState::SUCCEED);
        } else if (ws->GetTensor(operator_def_->input(i))->dim_size() == 2) {
          CONDITIONS(TransformFilter<T>(context,
                                        operator_def_.get(),
                                        i,
                                        OpenCLBufferType::BUFFER_2_BUFFER,
                                        mem_type,
                                        0,
                                        pruning_type_) == VanState::SUCCEED);
        } else if (ws->GetTensor(operator_def_->input(i))->dim_size() == 3) {
          CONDITIONS(TransformFilter<T>(context,
                                        operator_def_.get(),
                                        i,
                                        OpenCLBufferType::BUFFER_2_BUFFER,
                                        mem_type,
                                        0,
                                        pruning_type_) == VanState::SUCCEED);
        }
        // @JPEG - AI
        // else if (ws->GetTensor(operator_def_->input(i))->dim_size() == 5)
        // {
        //   // C3D CHECKME @vgod
        //   CONDITIONS(TransformFilterC3D<T>(context,
        //                                    operator_def_.get(),
        //                                    i,
        //                                    OpenCLBufferType::IN_OUT_CHANNEL,
        //                                    mem_type,
        //                                    0,
        //                                    pruning_type_) ==
        //                                    VanState::SUCCEED);
        // }
        else if (ws->GetTensor(operator_def_->input(i))->dim_size() == 0) {
          ws->GetTensor(operator_def_->input(i))->Reshape({1});
          CONDITIONS(TransformFilter<T>(context,
                                        operator_def_.get(),
                                        i,
                                        OpenCLBufferType::ARGUMENT,
                                        mem_type,
                                        0,
                                        pruning_type_) == VanState::SUCCEED);
        } else {
          UNSUPPORTED_OP("Elewise");
        }
      }
    }
  }
  VanState Run(OpContext *context) override {
    const Tensor *input0 = this->Input(0);
    const Tensor *input1 = this->InputSize() == 2 ? this->Input(1) : nullptr;
    Tensor *output = this->Output(0);
    return kernel_->Compute(context, input0, input1, output);
  }

private:
  std::unique_ptr<OpenCLEltwiseKernel> kernel_;
};
#endif // OPENCL_SUPPORT

void RegisterEltwise(OpRegistryBase *op_registry) {
  VAN_REGISTER_OP(op_registry, "Eltwise", EltwiseOp, DeviceType::CPU, float);
  VAN_REGISTER_OP(op_registry, "Eltwise", EltwiseOp, DeviceType::CPU, int32_t);

#ifdef OPENCL_SUPPORT
  VAN_REGISTER_OP(op_registry, "Eltwise", EltwiseOp, DeviceType::GPU, float);
  VAN_REGISTER_OP(op_registry, "Eltwise", EltwiseOp, DeviceType::GPU, half);
#endif // OPENCL_SUPPORT

  op_registry->Register(OpConditionBuilder("Eltwise").SetDevicePlacerFunc(
      [](OpConstructContext *context) -> std::set<DeviceType> {
        std::set<DeviceType> result;
        auto op_def = context->operator_def();
        auto executing_on = ProtoArgHelper::GetRepeatedArgs<OperatorProto, int>(
            *op_def,
            "executing_on",
            {static_cast<int>(DeviceType::CPU),
             static_cast<int>(DeviceType::GPU)});
        for (auto exe : executing_on) {
          result.insert(static_cast<DeviceType>(exe));
        }
        return result;
      }));
}

} // namespace deepvan
