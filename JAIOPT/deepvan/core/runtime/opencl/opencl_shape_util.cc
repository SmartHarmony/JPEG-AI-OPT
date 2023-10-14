#include "deepvan/core/runtime/opencl/opencl_shape_util.h"

#include <numeric>
#include <utility>

#include "deepvan/utils/logging.h"
#include "deepvan/utils/math.h"

namespace deepvan {
namespace {
// [(C + 3) / 4 * W, N * H]
void CalBertUniformImageShape(const std::vector<index_t> &shape,
                              std::vector<size_t> *image_shape) {
  CONDITIONS(shape.size() > 1);
  image_shape->resize(2);
  size_t shape_rank = shape.size();
  (*image_shape)[0] = RoundUpDiv4(shape[shape_rank - 1]);
  (*image_shape)[1] = std::accumulate(
      shape.cbegin(), shape.cend() - 1, 1, std::multiplies<size_t>());
}

// [(C + 3) / 4 * W, N * H]
void CalInOutputImageShape(const std::vector<index_t> &shape, /* NHWC */
                           std::vector<size_t> *image_shape) {
  CONDITIONS(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = RoundUpDiv4(shape[3]) * shape[2];
  (*image_shape)[1] = shape[0] * shape[1];
}

// NCDHW -> [(C + 3) / 4 * W, D * H]
void Cal3dInOutputImageShape(const std::vector<index_t> &shape, /* NCTHW */
                             std::vector<size_t> *image_shape) {
  CONDITIONS(shape.size() == 5);
  image_shape->resize(2);
  (*image_shape)[0] = RoundUpDiv4(shape[1]) * shape[4];
  (*image_shape)[1] = shape[0] * shape[2] * shape[3];
}

// [Ic, H * W * (Oc + 3) / 4]
void CalConv2dFilterImageShape(const std::vector<index_t> &shape, /* OIHW */
                               std::vector<size_t> *image_shape) {
  CONDITIONS(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = shape[1];
  (*image_shape)[1] = shape[2] * shape[3] * RoundUpDiv4(shape[0]);
}

// OIDHW -> [Ic, H * W * D * (Oc + 3) / 4]
void CalConv3dFilterImageShape(const std::vector<index_t> &shape, /* OITHW */
                               std::vector<size_t> *image_shape) {
  CONDITIONS(shape.size() == 5);
  image_shape->resize(2);
  (*image_shape)[0] = shape[1];
  (*image_shape)[1] = shape[2] * shape[3] * shape[4] * RoundUpDiv4(shape[0]);
}

// [H * W * M, (Ic + 3) / 4]
void CalDepthwiseConv2dFilterImageShape(
    const std::vector<index_t> &shape, /* MIHW */
    std::vector<size_t> *image_shape) {
  CONDITIONS(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = shape[0] * shape[2] * shape[3];
  (*image_shape)[1] = RoundUpDiv4(shape[1]);
}

void CalMatMulFilterImageShape(const std::vector<index_t> &shape,
                               std::vector<size_t> *image_shape) {
  CONDITIONS(shape.size() == 4 || shape.size() == 2);
  image_shape->resize(2);
  (*image_shape)[0] = RoundUpDiv4(shape[1]);
  (*image_shape)[1] = shape[0];
}

// [(size + 3) / 4, 1]
void CalArgImageShape(const std::vector<index_t> &shape,
                      std::vector<size_t> *image_shape) {
  CONDITIONS(shape.size() == 1);
  image_shape->resize(2);
  (*image_shape)[0] = RoundUpDiv4(shape[0]);
  (*image_shape)[1] = 1;
}

// Only support 3x3 now
// [ (Ic + 3) / 4, 16 * Oc]
void CalWinogradFilterImageShape(
    const std::vector<index_t> &shape, /* Oc, Ic, H, W*/
    std::vector<size_t> *image_shape,
    const int blk_size) {
  CONDITIONS(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = RoundUpDiv4(shape[1]);
  (*image_shape)[1] = (shape[0] * (blk_size + 2) * (blk_size + 2));
}

// [W * C, N * RoundUp<4>(H)]
void CalInOutHeightImageShape(const std::vector<index_t> &shape, /* NHWC */
                              std::vector<size_t> *image_shape) {
  CONDITIONS(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = shape[2] * shape[3];
  (*image_shape)[1] = shape[0] * RoundUpDiv4(shape[1]);
}

// [RoundUp<4>(W) * C, N * H]
void CalInOutWidthImageShape(const std::vector<index_t> &shape, /* NHWC */
                             std::vector<size_t> *image_shape) {
  CONDITIONS(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = RoundUpDiv4(shape[2]) * shape[3];
  (*image_shape)[1] = shape[0] * shape[1];
}

// [Ic * H * W, (Oc + 3) / 4]
void CalWeightHeightImageShape(const std::vector<index_t> &shape, /* OIHW */
                               std::vector<size_t> *image_shape) {
  CONDITIONS(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = shape[1] * shape[2] * shape[3];
  (*image_shape)[1] = RoundUpDiv4(shape[0]);
}

// [(Ic + 3) / 4 * H * W, Oc]
void CalWeightWidthImageShape(const std::vector<index_t> &shape, /* OIHW */
                              std::vector<size_t> *image_shape) {
  CONDITIONS(shape.size() == 4);
  image_shape->resize(2);
  (*image_shape)[0] = RoundUpDiv4(shape[1]) * shape[2] * shape[3];
  (*image_shape)[1] = shape[0];
}
} // namespace

void PatternUtil::CalImage2DShape(const std::vector<index_t> &shape,
                                  const OpenCLBufferType type,
                                  std::vector<size_t> &image_shape) {
  if (image_shape.empty()) {
    image_shape.resize(2);
  }
  switch (type) {
  case CONV2D_FILTER:
    //[ic, oc]
    image_shape[0] = static_cast<size_t>(shape[0]);
    image_shape[1] = static_cast<size_t>(shape[1]);
    break;
  case DW_CONV2D_FILTER: break;
  case IN_OUT_CHANNEL:
    //[(w + 3) / 4 * c, h]
    image_shape[0] = static_cast<size_t>(RoundUpDiv4(shape[3]) * shape[1]);
    image_shape[1] = static_cast<size_t>(shape[2]);
    break;
  case ARGUMENT:
    image_shape[0] = RoundUpDiv4(shape[0]);
    image_shape[1] = 1;
    break;
  case IN_OUT_HEIGHT: break;
  case IN_OUT_WIDTH: break;
  default: LOG(FATAL) << "Deepvan not supported yet.";
  }
}

void PatternUtil::CalImagePadShape(const std::vector<index_t> &original_shape,
                                   const std::vector<int> pads,
                                   std::vector<size_t> &padded_shape) {
  int max_width = std::max(RoundUpDiv4(original_shape[3]) + 1,
                           RoundUpDiv4(original_shape[3] + pads[0]));
  padded_shape[0] = (RoundUpDiv4(original_shape[3]) + 1) * original_shape[1];
  padded_shape[1] = original_shape[2] + pads[1];
}

void OpenCLUtil::CalImage2DShape(const std::vector<index_t> &shape, /* NHWC */
                                 const OpenCLBufferType type,
                                 std::vector<size_t> *image_shape,
                                 const int wino_block_size,
                                 ModelType model_type) {
  if (model_type == ModelType::BERT) {
    return CalBertShape(shape, type, image_shape, wino_block_size);
  }
  CONDITIONS_NOTNULL(image_shape);
  if (shape.size() == 5) {
    switch (type) {
    case CONV3D_FILTER: CalConv3dFilterImageShape(shape, image_shape); break;
    case IN_OUT_CHANNEL: Cal3dInOutputImageShape(shape, image_shape); break;
    default:
      LOG(FATAL) << "Deepvan not supported yet: "
                 << "shape: " << MakeString(shape) << ", type: " << type;
    }
  } else {
    switch (type) {
    case CONV2D_FILTER: CalConv2dFilterImageShape(shape, image_shape); break;
    case DW_CONV2D_FILTER: CalDepthwiseConv2dFilterImageShape(shape, image_shape); break;
    case IN_OUT_CHANNEL: CalInOutputImageShape(shape, image_shape); break;
    case ARGUMENT: CalArgImageShape(shape, image_shape); break;
    case IN_OUT_HEIGHT: CalInOutHeightImageShape(shape, image_shape); break;
    case IN_OUT_WIDTH: CalInOutWidthImageShape(shape, image_shape); break;
    case WINOGRAD_FILTER: CalWinogradFilterImageShape(shape, image_shape, wino_block_size); break;
    case WEIGHT_HEIGHT: CalWeightHeightImageShape(shape, image_shape); break;
    case WEIGHT_WIDTH: CalWeightWidthImageShape(shape, image_shape); break;
    case MATMUL_FILTER: CalMatMulFilterImageShape(shape, image_shape); break;
    case CONV2D_FILTER_BUFFER: CalConv2dFilterImageShape(shape, image_shape); break;
    default:
      LOG(FATAL) << "Deepvan not supported yet: "
                 << "shape: " << MakeString(shape) << ", type: " << type;
    }
  }
}

void OpenCLUtil::CalBertShape(const std::vector<index_t> &shape,
                              const OpenCLBufferType type,
                              std::vector<size_t> *image_shape,
                              const int wino_block_size) {
  CONDITIONS_NOTNULL(image_shape);
  if (shape.size() == 1) {
    image_shape->resize(2);
    switch (type) {
    case BUFFER_2_BUFFER:
    case IN_OUT_CHANNEL: CalArgImageShape(shape, image_shape); break;
    case ARGUMENT: CalArgImageShape(shape, image_shape); break;
    default:
      LOG(FATAL) << "Deepvan not supported yet: "
                 << "shape: " << MakeString(shape) << ", type: " << type
                 << ", wino_block_size: " << wino_block_size;
    }
  } else {
    CONDITIONS(shape.size() > 1);
    size_t shape_rank = shape.size();
    switch (type) {
    case MATMUL_FILTER:      
    case BUFFER_2_BUFFER:
    case IN_OUT_CHANNEL: CalBertUniformImageShape(shape, image_shape); break;
    default:
      LOG(FATAL) << "Deepvan not supported yet: "
                 << "shape: " << MakeString(shape) << ", type: " << type
                 << ", wino_block_size: " << wino_block_size;
    }
  }
}

void SliceUtil::CalImage3DShape(const std::vector<index_t> &shape,
                                const OpenCLBufferType type,
                                std::vector<size_t> *image_shape,
                                const int wino_blk_size) {
  CONDITIONS_NOTNULL(image_shape);
  if (shape.size() == 5) {
    switch (type) {
    case CONV3D_FILTER:
      image_shape->resize(2);
      (*image_shape)[0] =
          RoundUp<index_t>(shape[1], 4) * shape[2] * shape[3] * shape[4];
      (*image_shape)[1] = RoundUpDiv4(shape[0]);
      break;
    case IN_OUT_CHANNEL:
      image_shape->resize(2);
      (*image_shape)[0] = RoundUpDiv4(shape[4]) * shape[3];
      (*image_shape)[1] = shape[0] * shape[1] * shape[2];
      break;
    case WINOGRAD_FILTER:
      image_shape->resize(2);
      (*image_shape)[0] = shape[1];
      (*image_shape)[1] =
          (wino_blk_size + 2) * (wino_blk_size + 2) * RoundUpDiv4(shape[0]);
      break;
    default:
      LOG(FATAL) << "Deepvan not supported yet: "
                 << ", shape: " << MakeString(shape) << ", type: " << type;
    }
  } else if (shape.size() == 4) {
    switch (type) {
    case CONV2D_FILTER:
      image_shape->resize(2);
      (*image_shape)[0] = RoundUp<index_t>(shape[1], 4) * shape[2] * shape[3];
      (*image_shape)[1] = RoundUpDiv4(shape[0]);
      break;
    case IN_OUT_CHANNEL:
      image_shape->resize(2);
      (*image_shape)[0] = RoundUpDiv4(shape[3]) * shape[2];
      (*image_shape)[1] = shape[0] * shape[1];
      break;
    case WINOGRAD_FILTER:
      image_shape->resize(2);
      (*image_shape)[0] = shape[1];
      (*image_shape)[1] =
          (wino_blk_size + 2) * (wino_blk_size + 2) * RoundUpDiv4(shape[0]);
      break;
    default:
      LOG(FATAL) << "Deepvan not supported yet: "
                 << ", shape: " << MakeString(shape) << ", type: " << type;
    }
  } else {
    LOG(WARNING) << "Check me please!";
    switch (type) {
    case CONV2D_FILTER: CalConv2dFilterImageShape(shape, image_shape); break;
    case DW_CONV2D_FILTER:
      CalDepthwiseConv2dFilterImageShape(shape, image_shape);
      break;
    case IN_OUT_CHANNEL: CalInOutputImageShape(shape, image_shape); break;
    case ARGUMENT: CalArgImageShape(shape, image_shape); break;
    case IN_OUT_HEIGHT: CalInOutHeightImageShape(shape, image_shape); break;
    case IN_OUT_WIDTH: CalInOutWidthImageShape(shape, image_shape); break;
    case WINOGRAD_FILTER:
      CalWinogradFilterImageShape(shape, image_shape, wino_blk_size);
      break;
    case WEIGHT_HEIGHT: CalWeightHeightImageShape(shape, image_shape); break;
    case WEIGHT_WIDTH: CalWeightWidthImageShape(shape, image_shape); break;
    case MATMUL_FILTER: CalMatMulFilterImageShape(shape, image_shape); break;
    default:
      LOG(FATAL) << "Deepvan not supported yet: "
                 << "shape: " << MakeString(shape) << ", type: " << type;
    }
  }
}

std::shared_ptr<OperatorProto> OpenCLUtil::CreateTransformOpDef(
    const std::string &input_name,
    const std::vector<deepvan::index_t> &input_shape,
    const std::string &output_name,
    const deepvan::DataType dt,
    const OpenCLBufferType buffer_type,
    const deepvan::MemoryType mem_type,
    bool has_data_format,
    const PruningType pruning_type,
    const ModelType model_type) {
  std::unique_ptr<OperatorProto> op(new OperatorProto);
  std::string op_name = "deepvan_node_" + output_name;
  op->set_name(op_name);
  op->set_type("BufferTransform");
  op->add_input(input_name);
  op->add_output(output_name);
  op->set_pruning_type(pruning_type);
  op->set_model_type(model_type);
  ArgumentProto *arg = op->add_arg();
  arg->set_name("buffer_type");
  arg->set_i(static_cast<int32_t>(buffer_type));
  arg = op->add_arg();
  arg->set_name("mem_type");
  arg->set_i(static_cast<int32_t>(mem_type));
  arg = op->add_arg();
  arg->set_name("T");
  arg->set_i(static_cast<int32_t>(dt));
  arg = op->add_arg();
  arg->set_name("has_data_format");
  arg->set_i(has_data_format);
  if (!input_shape.empty()) {
    OutputShape *shape = op->add_output_shape();
    for (auto value : input_shape) {
      shape->add_dims(value);
    }
  }
  return std::move(op);
}

index_t ColumnUtil::CalBuffer2DSize(const std::vector<index_t> &shape,
                                    size_t ds,
                                    int align_size) {
  int rank = shape.size();
  index_t op_mem_size = 0;
  index_t ch_size = 0;
  index_t hw_size = 0;
  switch (rank) {
  case 2:
    op_mem_size = std::accumulate(
        shape.begin(), shape.end(), ds, std::multiplies<index_t>());
    op_mem_size = RoundUp<index_t>(op_mem_size, align_size);
    break;
  case 4:
    ch_size = std::accumulate(
        shape.begin(), shape.begin() + 2, ds, std::multiplies<index_t>());
    hw_size = std::accumulate(
        shape.begin() + 2, shape.end(), ds, std::multiplies<index_t>());
    op_mem_size = RoundUp<index_t>(ch_size, align_size) *
                  RoundUp<index_t>(hw_size, align_size);
    break;
  default:
    op_mem_size = std::accumulate(
        shape.begin(), shape.end(), ds, std::multiplies<index_t>());
    break;
  }
  return op_mem_size;
}

index_t ColumnUtil::CalIm2Col2DSize(const std::vector<index_t> &shape,
                                    int nnz,
                                    size_t ds,
                                    int align_size) {
  int rank = shape.size();
  index_t op_mem_size = 0;
  index_t hw_size = 0;
  switch (rank) {
  case 2: STUB; break;
  case 4:
    hw_size = std::accumulate(
        shape.begin() + 2, shape.end(), ds, std::multiplies<index_t>());
    op_mem_size = RoundUp<index_t>(nnz, align_size) *
                  RoundUp<index_t>(hw_size, align_size);
    break;
  default: STUB; break;
  }
  return op_mem_size;
}

} // namespace deepvan
