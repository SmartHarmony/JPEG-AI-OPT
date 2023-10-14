#ifndef DEEPVAN_BACKEND_ACTIVATION_OP_H_
#define DEEPVAN_BACKEND_ACTIVATION_OP_H_

#include <algorithm>
#include <cmath>

#include "deepvan/backend/common/activation_type.h"
#include "deepvan/core/types.h"

namespace deepvan {

template <typename T>
void DoActivation(const T *input_ptr,
                  T *output_ptr,
                  const index_t size,
                  const ActivationType type,
                  const float relux_max_limit,
                  const float leakyrelu_coefficient) {
  CONDITIONS(DataTypeToEnum<T>::value != DataType::DT_HALF);

  switch (type) {
  case NOOP: break;
  case RELU:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output_ptr[i] = std::max(input_ptr[i], static_cast<T>(0));
    }
    break;
  case RELUX:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output_ptr[i] = std::min(std::max(input_ptr[i], static_cast<T>(0)),
                               static_cast<T>(relux_max_limit));
    }
    break;
  case TANH:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output_ptr[i] = std::tanh(input_ptr[i]);
    }
    break;
  case SIGMOID:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output_ptr[i] = 1 / (1 + std::exp(-input_ptr[i]));
    }
    break;
  case LEAKYRELU:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output_ptr[i] =
          std::max(input_ptr[i], static_cast<T>(0)) +
          leakyrelu_coefficient * std::min(input_ptr[i], static_cast<T>(0));
    }
    break;
  default: LOG(FATAL) << "Unknown activation type: " << type;
  }
}

template <>
inline void DoActivation(const float *input_ptr,
                         float *output_ptr,
                         const index_t size,
                         const ActivationType type,
                         const float relux_max_limit,
                         const float leakyrelu_coefficient) {
  switch (type) {
  case NOOP: break;

  case TANH:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output_ptr[i] = std::tanh(input_ptr[i]);
    }
    break;
  case SIGMOID:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output_ptr[i] = 1 / (1 + std::exp(-input_ptr[i]));
    }
    break;
  case HARDSIGMOID:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output_ptr[i] = std::max(
          0.f,
          std::min(1.f,
                   relux_max_limit * input_ptr[i] + leakyrelu_coefficient));
    }
    break;
  case COS:
#pragma omp parallel for schedule(runtime)
    for (index_t i = 0; i < size; ++i) {
      output_ptr[i] = cos(input_ptr[i]);
    }
  default: LOG(FATAL) << "Unknown activation type: " << type;
  }
}

template <typename T>
void PReLUActivation(const T *input_ptr,
                     const index_t outer_size,
                     const index_t input_chan,
                     const index_t inner_size,
                     const T *alpha_ptr,
                     T *output_ptr) {
#pragma omp parallel for collapse(3) schedule(runtime)
  for (index_t i = 0; i < outer_size; ++i) {
    for (index_t chan_idx = 0; chan_idx < input_chan; ++chan_idx) {
      for (index_t j = 0; j < inner_size; ++j) {
        index_t idx = i * input_chan * inner_size + chan_idx * inner_size + j;
        if (input_ptr[idx] < 0) {
          output_ptr[idx] = input_ptr[idx] * alpha_ptr[chan_idx];
        } else {
          output_ptr[idx] = input_ptr[idx];
        }
      }
    }
  }
}

} // namespace deepvan

#endif // DEEPVAN_BACKEND_ACTIVATION_OP_H_
