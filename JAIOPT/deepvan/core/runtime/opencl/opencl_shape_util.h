#ifndef DEEPVAN_CORE_RUNTIME_OPENCL_OPENCL_UTIL_H_
#define DEEPVAN_CORE_RUNTIME_OPENCL_OPENCL_UTIL_H_

#include <memory>
#include <string>
#include <vector>

#include "deepvan/core/types.h"
#include "deepvan/export/deepvan.h"
#include "deepvan/utils/macros.h"
#include "deepvan/proto/deepvan.pb.h"

namespace deepvan {
enum OpenCLBufferType {
  CONV2D_FILTER = 0,
  IN_OUT_CHANNEL = 1,
  ARGUMENT = 2,
  IN_OUT_HEIGHT = 3,
  IN_OUT_WIDTH = 4,
  WINOGRAD_FILTER = 5,
  DW_CONV2D_FILTER = 6,
  WEIGHT_HEIGHT = 7,
  WEIGHT_WIDTH = 8,
  MATMUL_FILTER = 9,
  BUFFER_2_BUFFER = 10,
  CONV3D_FILTER = 11,
  CONV2D_FILTER_BUFFER = 12,
};

class OpenCLUtil {
public:
  static void CalImage2DShape(const std::vector<index_t> &shape, /* NHWC */
                              const OpenCLBufferType type,
                              std::vector<size_t> *image_shape,
                              const int wino_blk_size = 2,
                              ModelType model_type=ModelType::DEFAULT);

  static void CalBertShape(const std::vector<index_t> &shape, /* NHWC */
                           const OpenCLBufferType type,
                           std::vector<size_t> *image_shape,
                           const int wino_blk_size = 2);

  static std::shared_ptr<OperatorProto> CreateTransformOpDef(
      const std::string &input_name,
      const std::vector<deepvan::index_t> &input_shape,
      const std::string &output_name,
      const deepvan::DataType dt,
      const OpenCLBufferType buffer_type,
      const MemoryType mem_type,
      bool has_data_format,
      const PruningType pruning_type = PruningType::DENSE,
      const ModelType model_type = ModelType::DEFAULT);
};

class PatternUtil {
public:
  static void CalImage2DShape(const std::vector<index_t> &shape,
                              const OpenCLBufferType type,
                              std::vector<size_t> &image_shape);

  static void CalImagePadShape(const std::vector<index_t> &original_shape,
                               const std::vector<int> pads,
                               std::vector<size_t> &padded_shape);
};

class SliceUtil {
public:
  static void CalImage3DShape(const std::vector<index_t> &shape,
                              const OpenCLBufferType type,
                              std::vector<size_t> *image_shape,
                              const int wino_blk_size = 2);
};

class ColumnUtil {
public:
  static index_t CalBuffer2DSize(const std::vector<index_t> &shape,
                                 size_t ds,
                                 int align_size = 8);

  static index_t CalIm2Col2DSize(const std::vector<index_t> &shape,
                                 int nnz,
                                 size_t ds,
                                 int align_size = 8);

private:
  DISABLE_COPY_AND_ASSIGN(ColumnUtil);
};
} // namespace deepvan
#endif // DEEPVAN_CORE_RUNTIME_OPENCL_OPENCL_UTIL_H_
