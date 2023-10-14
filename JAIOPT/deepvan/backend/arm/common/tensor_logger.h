#ifndef DEEPVAN_BACKEND_ARM_COMMON_TENSOR_LOGGER_H_
#define DEEPVAN_BACKEND_ARM_COMMON_TENSOR_LOGGER_H_

#include <fstream>
#include <iostream>

#include "deepvan/core/tensor.h"

#define VECTOR_STRING(v) v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3]

namespace deepvan {
namespace debugger {

template <typename T>
inline void Show1DResult(const T *data,
                         const char *message = "",
                         int length = 10) {
  std::stringstream ss;
  ss << "=====Show 1D data info: " << message << "\n";
  for (int i = 0; i < length; i++) {
    ss << data[i] << ", ";
  }
  ss << "\n====================End for print====================\n";
  LOG(INFO) << ss.str();
}

// REMOVEME or DISABLEME when release
inline void Show4DResult(const Tensor *output,
                         const char *message = "",
                         int h_end = 3,
                         int w_end = 5) {
  auto channels = output->dim(1);
  auto height = output->dim(2);
  auto width = output->dim(3);
  h_end = h_end <= 0 ? height : h_end;
  w_end = w_end <= 0 ? width : w_end;
  auto image_size = height * width;
  auto output_data = output->data<float>();
  std::stringstream ss;
  ss << "=====Show Tensor info: " << MakeString(output->shape()) << "=====";
  ss << "Output for ";
  if (strlen(message) == 0) {
    ss << output->name();
  } else {
    ss << message;
  }
  ss << "\n";
  for (int c = 0; c < channels; c++) {
    auto output_data_ptr = output_data + c * image_size;
    ss << c << ", offset=" << c * image_size << ": ";
    for (int h = 0; h < h_end; h++) {
      for (int w = 0; w < w_end; w++) {
        ss << output_data_ptr[h * width + w] << ", ";
      }
      if (h != h_end - 1) {
        ss << " | ";
      }
    }
    ss << "\n";
  }
  ss << "====================End for print Output result====================";
  LOG(INFO) << ss.str();
}

inline void Print5DTensor(const Tensor *tensor, const char *message) {
  auto shape = tensor->shape();
  auto data = tensor->data<float>();
  std::stringstream ss;
  ss << message << " and shape: " << MakeString(shape) << "\n";
  index_t max_channel = std::min<index_t>(shape[1], 3);
  index_t max_series = std::min<index_t>(shape[2], 5);
  index_t max_height = std::min<index_t>(shape[3], 5);
  index_t max_width = std::min<index_t>(shape[4], 5);
  for (index_t c = 0; c < max_channel; c++) {
    for (index_t t = 0; t < max_series; t++) {
      for (index_t h = 0; h < max_height; h++) {
        ss << c << "\t" << t << "\t" << h << "\t";
        for (index_t w = 0; w < max_width; w++) {
          size_t offset = c * shape[2] * shape[3] * shape[4] +
                          t * shape[3] * shape[4] + h * shape[4] + w;
          ss << data[offset] << ", ";
        }
        ss << "\n";
      }
    }
  }
  LOG(INFO) << ss.str();
}

inline void Print2DTensor(const Tensor *tensor,
                          const char *message,
                          int max = 20) {
  auto shape = tensor->shape();
  auto data = tensor->data<float>();
  std::stringstream ss;
  ss << message << "\n";
  index_t max_height = std::min<index_t>(shape[0], 1);
  index_t max_width = std::min<index_t>(shape[1], max);
  for (index_t h = 0; h < max_height; h++) {
    for (index_t w = 0; w < max_width; w++) {
      size_t offset = h * shape[1] + w;
      ss << data[offset] << ", ";
    }
    ss << "\n";
  }
  LOG(INFO) << ss.str();
}

inline void WriteTensor2File(const Tensor *tensor, std::string name) {
  std::string output_name = "/data/local/tmp/deepvan_run/out_" + name;
  auto output_shape = tensor->shape();
  std::ofstream out_file(output_name, std::ios::binary);
  // only support float and int32
  int64_t output_size = std::accumulate(
      output_shape.begin(), output_shape.end(), 4, std::multiplies<int64_t>());
  out_file.write(tensor->data<char>(), output_size);
  out_file.flush();
  out_file.close();
  LOG(INFO) << "Write output file " << output_name << " with size "
            << output_size << "(" << MakeString(output_shape) << ") done.";
}

} // namespace debugger
} // namespace deepvan

#endif