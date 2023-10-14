#ifndef DEEPVAN_BACKEND_OPENCL_IMAGE_IMAGE_TENSOR_DEBUG_H_
#define DEEPVAN_BACKEND_OPENCL_IMAGE_IMAGE_TENSOR_DEBUG_H_

#include "deepvan/core/tensor.h"

namespace deepvan {
namespace opencl {
namespace image {

// REMOVEME or DISABLEME when release
inline void Show4DResult(const Tensor *output,
                         std::vector<size_t> &output_pitch,
                         const char *message = "",
                         int h_end = 3,
                         int w_end = 5,
                         bool nhwc = true) {
  auto height = output->dim(1);
  auto width = output->dim(2);
  auto channels = output->dim(3);
  auto data = output->data<float>();
  h_end = h_end <= 0 ? height : h_end;
  w_end = w_end <= 0 ? width : w_end;
  std::stringstream ss;
  ss << "=====Show Tensor info: " << MakeString(output->shape()) << "=====";
  ss << "Output for ";
  if (strlen(message) == 0) {
    ss << output->name();
  } else {
    ss << message;
  }
  ss << "\n";
  int stride =
      static_cast<int>(output_pitch[0]) / GetEnumTypeSize(output->dtype());
  if (nhwc) {
    for (int h = 0; h < h_end; h++) {
      ss << h << ": ";
      for (int w = 0; w < w_end; w++) {
        for (int c = 0; c < channels; c++) {
          int ch_blk = c / 4;
          ss << data[h * stride + ch_blk * 4 * width + w * 4 + c % 4] << ", ";
        }
      }
      ss << "\n";
    }
  } else {
    for (int c = 0; c < channels; c++) {
      ss << c << ": ";
      int ch_blk = c / 4;
      for (int h = 0; h < h_end; h++) {
        for (int w = 0; w < w_end; w++) {
          ss << data[h * stride + ch_blk * 4 * width + w * 4 + c % 4] << ", ";
        }
        ss << " | ";
      }
      ss << "\n";
    }
  }
  ss << "====================End for print Output result====================";
  LOG(INFO) << ss.str();
}

template <typename T>
inline void Show4DResult(const Tensor *output,
                         std::vector<size_t> &output_pitch,
                         const char *message = "",
                         int h_end = 3,
                         int w_end = 5,
                         bool nhwc = true) {
  auto height = output->dim(1);
  auto width = output->dim(2);
  auto channels = output->dim(3);
  auto data = output->data<T>();
  h_end = h_end <= 0 ? height : h_end;
  w_end = w_end <= 0 ? width : w_end;
  std::stringstream ss;
  ss << "=====Show Tensor info: " << MakeString(output->shape()) << "=====";
  ss << "Output for ";
  if (strlen(message) == 0) {
    ss << output->name();
  } else {
    ss << message;
  }
  ss << "\n";
  int stride =
      static_cast<int>(output_pitch[0]) / GetEnumTypeSize(output->dtype());
  if (nhwc) {
    for (int h = 0; h < h_end; h++) {
      ss << h << ": ";
      for (int w = 0; w < w_end; w++) {
        for (int c = 0; c < channels; c++) {
          int ch_blk = c / 4;
          ss << data[h * stride + ch_blk * 4 * width + w * 4 + c % 4] << ", ";
        }
      }
      ss << "\n";
    }
  } else {
    for (int c = 0; c < channels; c++) {
      ss << c << ": ";
      int ch_blk = c / 4;
      for (int h = 0; h < h_end; h++) {
        for (int w = 0; w < w_end; w++) {
          ss << data[h * stride + ch_blk * 4 * width + w * 4 + c % 4] << ", ";
        }
        ss << " | ";
      }
      ss << "\n";
    }
  }
  ss << "====================End for print Output result====================";
  LOG(INFO) << ss.str();
}

template <typename T>
inline void Show5DResult(const Tensor *output,
                         std::vector<size_t> &output_pitch,
                         const char *message = "",
                         int c_end = 3,
                         int d_end = 3,
                         int h_end = 3,
                         int w_end = 3) {
  auto channels = output->dim(1);
  auto depth = output->dim(2);
  auto height = output->dim(3);
  auto width = output->dim(4);
  auto data = output->data<T>();
  c_end = c_end <= 0 ? channels : std::min<int>(c_end, channels);
  d_end = d_end <= 0 ? depth : std::min<int>(d_end, depth);
  h_end = h_end <= 0 ? height : std::min<int>(h_end, height);
  w_end = w_end <= 0 ? width : std::min<int>(w_end, width);
  std::stringstream ss;
  ss << "=====Show Tensor info: " << MakeString(output->shape()) << "=====";
  ss << "Output for ";
  if (strlen(message) == 0) {
    ss << output->name();
  } else {
    ss << message;
  }
  ss << "\n";
  int stride =
      static_cast<int>(output_pitch[0]) / GetEnumTypeSize(output->dtype());
  for (int c = 0; c < c_end; c++) {
    int ch_blk = c / 4;
    for (int d = 0; d < d_end; d++) {
      for (int h = 0; h < h_end; h++) {
        ss << c << "\t" << d << "\t" << h << "\t";
        for (int w = 0; w < w_end; w++) {
          ss << data[(d * height + h) * stride + ch_blk * 4 * width + w * 4 +
                     c % 4];
          if (w != w_end - 1) {
            ss << ", ";
          }
        }
        ss << "\n";
      }
    }
  }
  ss << "====================End for print Output result====================";
  LOG(INFO) << ss.str();
} // namespace image

inline void WriteTensor2File(const Tensor *tensor,
                             std::vector<size_t> &output_pitch,
                             std::string name,
                             bool need_map = true) {
  if (need_map) {
    output_pitch.resize(2);
    tensor->UnderlyingBuffer()->Map(&output_pitch);
  }
  std::string output_name = "/data/local/tmp/deepvan_run/out_" + name;
  auto image_shape = tensor->image_shape();
  auto data = tensor->data<char>();
  std::ofstream out_file(output_name, std::ios::binary);
  int stride = static_cast<int>(output_pitch[0]);
  for (size_t h = 0; h < image_shape[1]; h++) {
    out_file.write(data, image_shape[0] * 4 * GetEnumTypeSize(tensor->dtype()));
    data += stride;
  }
  out_file.flush();
  out_file.close();
  LOG(INFO) << "Write output file " << output_name << " with image shape "
            << MakeString(image_shape) << "(" << MakeString(tensor->shape())
            << ") done.";
  if (need_map) {
    tensor->UnderlyingBuffer()->UnMap();
  }
}

template <typename T>
inline void WriteTensorToFile(const Tensor *output,
                              std::vector<size_t> &output_pitch,
                              std::string name,
                              bool need_map = true) {
  if (need_map) {
    output_pitch.resize(2);
    output->UnderlyingBuffer()->Map(&output_pitch);
  }
  auto image_shape = output->image_shape();
  auto data = output->data<char>();
  auto channels = output->dim(4);
  auto depth = output->dim(1);
  auto height = output->dim(2);
  auto width = output->dim(3);
  auto rounded_channels = RoundUp<index_t>(channels, 4);

  std::string output_name = "/data/local/tmp/deepvan_run/out_" + name;
  std::ofstream out_file(output_name, std::ios::binary);
  int type_size = GetEnumTypeSize(output->dtype());
  int stride = static_cast<int>(output_pitch[0] / type_size);
  for (index_t d = 0; d < depth; d++) {
    for (index_t h = 0; h < height; h++) {
      for (index_t w = 0; w < width; w++) {
        for (index_t c = 0; c < channels; c++) {
          index_t offset = (d * height + h) * stride + w * rounded_channels + c;
          auto data_ptr = data + offset * type_size;
          out_file.write(data_ptr, type_size);
        }
      } // w
    }   // h
  }     // d
  out_file.flush();
  out_file.close();
  LOG(INFO) << "Write output file " << output_name << " with image shape "
            << MakeString(image_shape) << "(" << MakeString(output->shape())
            << ") done.";
  if (need_map) {
    output->UnderlyingBuffer()->UnMap();
  }
}

template <typename T>
inline void Write4DTensorToFile(const Tensor *output,
                                std::vector<size_t> &output_pitch,
                                std::string name,
                                bool need_map = true) {
  if (need_map) {
    output_pitch.resize(2);
    output->UnderlyingBuffer()->Map(&output_pitch);
  }
  auto image_shape = output->image_shape();
  auto data = output->data<char>();
  auto height = output->dim(1);
  auto width = output->dim(2);
  auto channels = output->dim(3);
  auto rounded_channels = RoundUp<index_t>(channels, 4);

  std::string output_name = "/data/local/tmp/deepvan_run/out_" + name;
  std::ofstream out_file(output_name, std::ios::binary);
  int type_size = GetEnumTypeSize(output->dtype());
  int stride = static_cast<int>(output_pitch[0] / type_size);
  for (index_t h = 0; h < height; h++) {
    for (index_t w = 0; w < width; w++) {
      for (index_t c = 0; c < channels; c++) {
        index_t offset = h * stride + w * rounded_channels + c;
        auto data_ptr = data + offset * type_size;
        out_file.write(data_ptr, type_size);
      }
    } // w
  }   // h
  out_file.flush();
  out_file.close();
  LOG(INFO) << "Write output file " << output_name << " with image shape "
            << MakeString(image_shape) << "(" << MakeString(output->shape())
            << ") done.";
  if (need_map) {
    output->UnderlyingBuffer()->UnMap();
  }
}

template <typename T>
inline void WriteNHWC4DResult(const Tensor *output,
                              std::vector<size_t> &output_pitch,
                              std::string name,
                              bool need_map = true) {
  if (need_map) {
    output_pitch.resize(2);
    output->UnderlyingBuffer()->Map(&output_pitch);
  }
  auto image_shape = output->image_shape();
  auto data = output->data<char>();
  auto height = output->dim(1);
  auto width = output->dim(2);
  auto channels = output->dim(3);
  auto rounded_channels = RoundUp<index_t>(channels, 4);

  std::string output_name = "/data/local/tmp/deepvan_run/out_" + name;
  std::ofstream out_file(output_name, std::ios::binary);
  int type_size = GetEnumTypeSize(output->dtype());
  int stride = static_cast<int>(output_pitch[0] / type_size);
  for (index_t h = 0; h < height; h++) {
    for (index_t w = 0; w < width; w++) {
      for (index_t c = 0; c < channels; c++) {
        int ch_blk = c / 4;
        index_t offset = h * stride + ch_blk * 4 * width + w * 4 + c % 4;
        auto data_ptr = data + offset * type_size;
        out_file.write(data_ptr, type_size);
      }
    } // w
  }   // h
  out_file.flush();
  out_file.close();
  LOG(INFO) << "Write output file " << output_name << " with image shape "
            << MakeString(image_shape) << "(" << MakeString(output->shape())
            << ") done.";
  if (need_map) {
    output->UnderlyingBuffer()->UnMap();
  } 
}

} // namespace image
} // namespace opencl
} // namespace deepvan

#endif // DEEPVAN_BACKEND_OPENCL_IMAGE_IMAGE_TENSOR_DEBUG_H_