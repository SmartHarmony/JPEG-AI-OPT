#ifndef DEEPVAN_UTILS_MATH_H_
#define DEEPVAN_UTILS_MATH_H_

#include <cmath>

#include <algorithm>
#include <vector>

#include "deepvan/utils/logging.h"

namespace deepvan {
template <typename Integer>
Integer RoundUp(Integer i, Integer factor) {
  return (i + factor - 1) / factor * factor;
}

template <typename Integer, uint32_t factor>
Integer RoundUpDiv(Integer i) {
  return (i + factor - 1) / factor;
}

// Partial specialization of function templates is not allowed
template <typename Integer>
Integer RoundUpDiv4(Integer i) {
  return (i + 3) >> 2;
}

template <typename Integer>
Integer RoundUpDiv8(Integer i) {
  return (i + 7) >> 3;
}

template <typename Integer>
Integer RoundUpDiv(Integer i, Integer factor) {
  return (i + factor - 1) / factor;
}

template <typename Integer>
Integer CeilQuotient(Integer a, Integer b) {
  return (a + b - 1) / b;
}

template <typename Integer>
inline Integer Clamp(Integer in, Integer low, Integer high) {
  return std::max<Integer>(low, std::min<Integer>(in, high));
}

inline float ScalarSigmoid(float in) {
  if (in > 0) {
    return 1 / (1 + std::exp(-in));
  } else {
    float x = std::exp(in);
    return x / (x + 1.f);
  }
}

inline float ScalarTanh(float in) {
  if (in > 0) {
    float x = std::exp(-in);
    return -1.f + 2.f / (1.f + x * x);
  } else {
    float x = std::exp(in);
    return 1.f - 2.f / (1.f + x * x);
  }
}

template <typename SrcType, typename DstType>
std::vector<DstType> TransposeShape(const std::vector<SrcType> &shape,
                                    const std::vector<int> &dst_dims) {
  size_t shape_dims = shape.size();
  CONDITIONS(shape_dims == dst_dims.size());
  std::vector<DstType> output_shape(shape_dims);
  for (size_t i = 0; i < shape_dims; ++i) {
    output_shape[i] = static_cast<DstType>(shape[dst_dims[i]]);
  }
  return output_shape;
}

template <typename T>
const T& clamp(const T& v, const T& lo, const T& hi) {
  return std::min<T>(std::max<T>(v, lo), hi);
}

template <typename T>
const T& min2(const T&v, const T& lo1, const T& lo2) {
  return std::min<T>(std::min<T>(v, lo1), lo2);
}

template <typename T>
const T& max2(const T&v, const T& lo1, const T& lo2) {
  return std::max<T>(std::max<T>(v, lo1), lo2);
}

template <typename T>
void IncreaseIndexInDims(int ndim, 
                         const T* dims, 
                         T* index) {
  for (int i = ndim - 1; i >= 0; --i) {
    ++index[i];
    if (index[i] >= dims[i]) {
      index[i] -= dims[i];
    } else {
      break;
    }
  }
}

template <typename T>
void ComputeTransposedStrides(int ndim, 
                              const T* dims, 
                              const int* axes, 
                              T* strides) {
  std::vector<T> buff(ndim);
  T cur_stride = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    buff[i] = cur_stride;
    cur_stride *= dims[i];
  }
  for (int i = 0; i < ndim; ++i) {
    strides[i] = buff[axes[i]];
  }
}


}  // namespace deepvan

#endif  // DEEPVAN_UTILS_MATH_H_
