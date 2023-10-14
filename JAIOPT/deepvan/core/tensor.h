#ifndef DEEPVAN_CORE_TENSOR_H_
#define DEEPVAN_CORE_TENSOR_H_

#include <algorithm>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

// #include "deepvan/core/block.h"
#include "deepvan/core/buffer.h"
// #include "deepvan/core/columns.h"
// #include "deepvan/core/csr.h"
// #include "deepvan/core/pattern.h"
#include "deepvan/core/preallocated_pooled_allocator.h"
// #include "deepvan/core/slice.h"
#include "deepvan/core/types.h"
#ifdef OPENCL_SUPPORT
#include "deepvan/core/runtime/opencl/cl2_header.h"
#endif
#include "deepvan/utils/logging.h"

#ifdef NEON_SUPPORT
// Avoid over-bound accessing memory
#define DEEPVAN_EXTRA_BUFFER_PAD_SIZE 64
#else
#define DEEPVAN_EXTRA_BUFFER_PAD_SIZE 0
#endif

namespace deepvan {
#define DEEPVAN_SINGLE_ARG(...) __VA_ARGS__
#define DEEPVAN_CASE(TYPE, STATEMENTS)                                         \
  case DataTypeToEnum<TYPE>::value: {                                          \
    typedef TYPE T;                                                            \
    STATEMENTS;                                                                \
    break;                                                                     \
  }

#ifdef OPENCL_SUPPORT
#define DEEPVAN_TYPE_ENUM_SWITCH(                                              \
    TYPE_ENUM, STATEMENTS, INVALID_STATEMENTS, DEFAULT_STATEMENTS)             \
  switch (TYPE_ENUM) {                                                         \
    DEEPVAN_CASE(half, DEEPVAN_SINGLE_ARG(STATEMENTS))                         \
    DEEPVAN_CASE(float, DEEPVAN_SINGLE_ARG(STATEMENTS))                        \
    DEEPVAN_CASE(uint8_t, DEEPVAN_SINGLE_ARG(STATEMENTS))                      \
    DEEPVAN_CASE(int32_t, DEEPVAN_SINGLE_ARG(STATEMENTS))                      \
  case DT_INVALID: INVALID_STATEMENTS; break;                                  \
  default: DEFAULT_STATEMENTS; break;                                          \
  }
#else
#define DEEPVAN_TYPE_ENUM_SWITCH(                                              \
    TYPE_ENUM, STATEMENTS, INVALID_STATEMENTS, DEFAULT_STATEMENTS)             \
  switch (TYPE_ENUM) {                                                         \
    DEEPVAN_CASE(float, DEEPVAN_SINGLE_ARG(STATEMENTS))                        \
    DEEPVAN_CASE(uint8_t, DEEPVAN_SINGLE_ARG(STATEMENTS))                      \
    DEEPVAN_CASE(int32_t, DEEPVAN_SINGLE_ARG(STATEMENTS))                      \
  case DT_INVALID: INVALID_STATEMENTS; break;                                  \
  default: DEFAULT_STATEMENTS; break;                                          \
  }
#endif

// `TYPE_ENUM` will be converted to template `T` in `STATEMENTS`
#define DEEPVAN_RUN_WITH_TYPE_ENUM(TYPE_ENUM, STATEMENTS)                      \
  DEEPVAN_TYPE_ENUM_SWITCH(                                                    \
      TYPE_ENUM, STATEMENTS, LOG(FATAL) << "Invalid type";                     \
      , LOG(FATAL) << "Unknown type: " << TYPE_ENUM;)

namespace numerical_chars {
inline std::ostream &operator<<(std::ostream &os, char c) {
  return std::is_signed<char>::value ? os << static_cast<int>(c)
                                     : os << static_cast<unsigned int>(c);
}

inline std::ostream &operator<<(std::ostream &os, signed char c) {
  return os << static_cast<int>(c);
}

inline std::ostream &operator<<(std::ostream &os, unsigned char c) {
  return os << static_cast<unsigned int>(c);
}
} // namespace numerical_chars

enum PatternDataType {
  DT_NONE = 0,
  DT_ORDER = 1,
  DT_FILTER_NUMS = 2,
  DT_FILTER_STYLE = 3,
  DT_FILTER_GAP = 4,
  DT_FILTER_INDEX = 5,
};
enum CSRDataType {
  DT_ROWPTR = 0,
  DT_COLPTR = 1,
};

enum class SliceDataType {
  DT_ORDER = 0,
  DT_INDEX,
  DT_OFFSET,
};

enum ColumnDataType {
  DT_ROWS = 0,
  DT_COLS = 1,
  DT_ZEROS = 2,
};

class Tensor {
public:
  Tensor(Allocator *alloc,
         DataType type,
         bool is_weight = false,
         const std::string name = "")
      : allocator_(alloc),
        dtype_(type),
        buffer_(nullptr),
        is_buffer_owner_(true),
        unused_(false),
        name_(name),
        is_weight_(is_weight),
        scale_(0.f),
        zero_point_(0),
        minval_(0.f),
        maxval_(0.f) {}

  Tensor(BufferBase *buffer,
         DataType dtype,
         bool is_weight = false,
         const std::string name = "")
      : dtype_(dtype),
        buffer_(buffer),
        is_buffer_owner_(false),
        unused_(false),
        name_(name),
        is_weight_(is_weight),
        scale_(0.f),
        zero_point_(0),
        minval_(0.f),
        maxval_(0.f) {}

  Tensor(const BufferSlice &buffer_slice,
         DataType dtype,
         bool is_weight = false,
         const std::string name = "")
      : dtype_(dtype),
        buffer_slice_(buffer_slice),
        is_buffer_owner_(false),
        unused_(false),
        name_(name),
        is_weight_(is_weight),
        scale_(0.f),
        zero_point_(0),
        minval_(0.f),
        maxval_(0.f) {
    buffer_ = &buffer_slice_;
  }

  explicit Tensor(bool is_weight = false)
      : Tensor(GetCPUAllocator(), DT_FLOAT, is_weight) {}

  ~Tensor() {
    if (is_buffer_owner_ && buffer_ != nullptr) {
      delete buffer_;
    }
  }

  inline std::string name() const { return name_; }

  inline DataType dtype() const { return dtype_; }

  inline void SetDtype(DataType dtype) { dtype_ = dtype; }

  inline bool unused() const { return unused_; }

  inline const std::vector<index_t> &shape() const { return shape_; }

  inline const std::vector<size_t> &pattern_shape() const {
    return pattern_shape_;
  }

  inline std::vector<index_t> max_shape() const {
    if (shape_configured_.empty()) {
      return shape();
    } else {
      auto &_shape = shape();
      std::vector<index_t> max_shape(_shape.size());
      CONDITIONS(_shape.size() == shape_configured_.size());
      for (size_t i = 0; i < shape_configured_.size(); ++i) {
        max_shape[i] = std::max(_shape[i], shape_configured_[i]);
      }
      return max_shape;
    }
  }

  inline index_t max_size() const {
    auto _max_shape = max_shape();
    return std::accumulate(
        _max_shape.begin(), _max_shape.end(), 1, std::multiplies<index_t>());
  }

  inline index_t raw_max_size() const { return max_size() * SizeOfType(); }

  inline void SetShapeConfigured(const std::vector<index_t> &shape_configured) {
    shape_configured_ = shape_configured;
  }

  inline const std::vector<index_t> &buffer_shape() const {

    return buffer_shape_;
  }

  inline index_t dim_size() const { return shape_.size(); }

  inline index_t dim(unsigned int index) const {
    CONDITIONS(index < shape_.size(),
               name_,
               ": Dim out of range: ",
               index,
               " >= ",
               shape_.size());
    return shape_[index];
  }

  // CHECKME @niuwei, when/where call this, why is_pattern_weight true sometimes
  inline index_t size() const {

    return std::accumulate(
        shape_.begin(), shape_.end(), 1, std::multiplies<int64_t>());
  }

  inline index_t raw_size() const { return size() * SizeOfType(); }

  inline const std::vector<size_t> &image_shape() const { return image_shape_; }

  inline bool has_opencl_image() const {
    return buffer_ != nullptr && !buffer_->OnHost() &&
           buffer_->buffer_type() == core::BufferType::BT_IMAGE;
  }

  inline bool has_opencl_buffer() const {
    return buffer_ != nullptr && !buffer_->OnHost() && !has_opencl_image();
  }

  inline bool has_opencl_image(const BufferBase *buf) const {
    return !buf->OnHost() && buf->buffer_type() == core::BufferType::BT_IMAGE;
  }

  inline bool has_opencl_buffer(const BufferBase *buf) const {
    return !buf->OnHost() && !has_opencl_image(buf);
  }

  inline MemoryType memory_type() const {
    CONDITIONS(buffer_ != nullptr, "Tensor ", name_, " is empty");
    BufferBase *buf = buffer_;
    if (buf->OnHost()) {
      return MemoryType::CPU_BUFFER;
    } else if (buf->buffer_type() == core::BufferType::BT_IMAGE) {
      return MemoryType::GPU_IMAGE;
    } else {
      return MemoryType::GPU_BUFFER;
    }
  }

  inline void set_data_format(DataFormat data_format) {
    data_format_ = data_format;
  }

  inline DataFormat data_format() const { return data_format_; }

#ifdef OPENCL_SUPPORT
  inline cl::Image *opencl_image() const {
    CONDITIONS(has_opencl_image(), name_, " do not have image");
    return static_cast<cl::Image *>(buffer_->buffer());
  }

  inline cl::Buffer *opencl_buffer() const {
    CONDITIONS(has_opencl_buffer(), name_, " do not have opencl buffer");
    return static_cast<cl::Buffer *>(buffer_->buffer());
  }

#endif

  inline index_t buffer_offset() const { return buffer_->offset(); }

  inline const void *raw_data() const {
    CONDITIONS(buffer_ != nullptr, "buffer is null");
    return buffer_->raw_data();
  }

  template <typename T>
  inline const T *data() const {
    CONDITIONS_NOTNULL(buffer_);
    return buffer_->data<T>();
  }

  inline void *raw_mutable_data() {
    CONDITIONS_NOTNULL(buffer_);
    return buffer_->raw_mutable_data();
  }

  template <typename T>
  inline T *mutable_data() {
    CONDITIONS_NOTNULL(buffer_);
    return static_cast<T *>(buffer_->raw_mutable_data());
  }

  template <typename T>
  inline float sum_of_tensor(std::string message = "") const {
    CONDITIONS_NOTNULL(buffer_);
    const T *data = static_cast<const T *>(buffer_->raw_data());
    float result = 0;
    for (index_t i = 0; i < size(); i++) {
      result += data[i];
    }
    if (message != "") {
      LOG(INFO) << "Sum of tensor " << message << " is: " << result;
    }
    return result;
  }

  template <typename T>
  inline const T *row_ptr_data() const {
    return static_cast<const T *>(sparsed_row_.raw_data());
  }

  template <typename T>
  inline T *mutable_row_ptr_data() {
    CONDITIONS_NOTNULL(buffer_);
    return static_cast<T *>(sparsed_row_.raw_mutable_data());
  }

  template <typename T>
  inline const T *col_index_data() const {
    return static_cast<const T *>(sparsed_col_.raw_data());
  }

  template <typename T>
  inline T *mutable_col_index_data() {
    CONDITIONS_NOTNULL(buffer_);
    return static_cast<T *>(sparsed_col_.raw_mutable_data());
  }

  inline void MarkUnused() { unused_ = true; }

  inline void Clear() {

    {
      CONDITIONS_NOTNULL(buffer_);
      buffer_->Clear(raw_size());
    }
  }

  inline void Reshape(const std::vector<index_t> &shape) {
    shape_ = shape;
    if (has_opencl_image()) {
      CONDITIONS(raw_size() <= 4 * buffer_->size(),
                 "Must satisfy: ",
                 raw_size(),
                 " <= ",
                 4 * buffer_->size());
    } else {
      CONDITIONS(raw_size() <= buffer_->size(),
                 "Must satisfy: ",
                 raw_size(),
                 " <= ",
                 buffer_->size());
    }
  }

  inline VanState Resize(const std::vector<index_t> &shape) {
    shape_ = shape;
    buffer_shape_ = shape;
    image_shape_.clear();
    if (buffer_ != nullptr) {
      CONDITIONS(!has_opencl_image(),
                 name_,
                 ": Cannot resize image, use ResizeImage.");
      const index_t apply_size =
          raw_size() +
          ((buffer_ != &buffer_slice_) ? DEEPVAN_EXTRA_BUFFER_PAD_SIZE : 0);
      if (apply_size > buffer_->size()) {
        LOG(WARNING) << name_ << ": Resize buffer from size " << buffer_->size()
                     << " to " << apply_size;
        CONDITIONS(buffer_ != &buffer_slice_,
                   ": Cannot resize tensor with buffer slice");
        return buffer_->Resize(apply_size);
      }
      return VanState::SUCCEED;
    } else {
      // if (shape_.size() == 4) {
      //   LOG(INFO) << DEBUG_GPU << "Raw size: " << raw_size()
      //             << ", CSR Weight: " << (is_csr_weight() && csr_data_)
      //             << ", Pattern Weight: "
      //             << (is_pattern_weight() && pattern_data_)
      //             << ", Column Weight: " << (is_column_weight() &&
      //             column_data_)
      //             << ", Shape: " << MakeString(shape_)
      //             << ", Buffer Shape: " << MakeString(buffer_shape_);
      // }
      CONDITIONS(is_buffer_owner_);
      buffer_ = new Buffer(allocator_);
      return buffer_->Allocate(raw_size() + DEEPVAN_EXTRA_BUFFER_PAD_SIZE);
    }
  }

  // Make this tensor reuse other tensor's buffer.
  // This tensor has the same dtype, shape and image_shape.
  // It could be reshaped later (with image shape unchanged).
  inline void ReuseTensorBuffer(const Tensor &other) {
    if (is_buffer_owner_ && buffer_ != nullptr) {
      delete buffer_;
    }
    is_buffer_owner_ = false;
    buffer_ = other.buffer_;
    allocator_ = other.allocator_;
    dtype_ = other.dtype_;
    shape_ = other.shape_;
    buffer_shape_ = other.buffer_shape_;
    image_shape_ = other.image_shape_;
  }

  inline VanState ResizeSparseImage(const Tensor &old_tensor) {
    shape_ = old_tensor.shape();
    buffer_shape_ = old_tensor.shape();
    pattern_shape_ = old_tensor.pattern_shape_;
    image_shape_ = old_tensor.image_shape_;
    if (buffer_ == nullptr) {
      CONDITIONS(is_buffer_owner_);
      buffer_ = new Image(allocator_);
      VLOG(1) << "Allocate OpenCL image: " << is_weight();
      return buffer_->Allocate(pattern_shape_, dtype_);
    } else {
      return VanState::SUCCEED;
    }
  }

  inline VanState ResizeImage(const std::vector<index_t> &shape,
                              const std::vector<size_t> &image_shape) {
    // WEI NIU
    shape_ = shape;
    buffer_shape_ = shape;
    image_shape_ = image_shape;
    if (buffer_ == nullptr) {
      CONDITIONS(is_buffer_owner_);
      buffer_ = new Image(allocator_);
      VLOG(1) << "Allocate OpenCL image: " << is_weight();
      return buffer_->Allocate(image_shape, dtype_);
    } else {
      CONDITIONS(has_opencl_image(),
                 name_,
                 ": Cannot ResizeImage buffer, use Resize.");
      CONDITIONS(image_shape[0] <= buffer_->shape()[0] &&
                     image_shape[1] <= buffer_->shape()[1],
                 "tensor (source op ",
                 name_,
                 "): current logical image shape:",
                 image_shape[0],
                 ", ",
                 image_shape[1],
                 " > physical image shape: ",
                 buffer_->shape()[0],
                 ", ",
                 buffer_->shape()[1],
                 ", with shape: ")
          << MakeString(shape_);
      VLOG(1) << "Only Resize OpenCL image: " << is_weight();
      return VanState::SUCCEED;
    }
  }

  inline VanState ResizeLike(const Tensor &other) { return ResizeLike(&other); }

  inline VanState ResizeLike(const Tensor *other) {
    if (other->has_opencl_image()) {
      if (is_buffer_owner_ && buffer_ != nullptr && !has_opencl_image()) {
        delete buffer_;
        buffer_ = nullptr;
      }
      return ResizeImage(other->shape(), other->image_shape_);
    } else {
      if (is_buffer_owner_ && buffer_ != nullptr && has_opencl_image()) {
        delete buffer_;
        buffer_ = nullptr;
      }
      return Resize(other->shape());
    }
  }

  inline void CopyBytes(const void *src, size_t size) {
    MappingGuard guard(this);
    memcpy(buffer_->raw_mutable_data(), src, size);
  }

  inline void CopyBytesWithMultiCore(const void *src, size_t size) {
    MappingGuard guard(this);
    size_t seg = 8;
    const size_t length = (size + seg - 1) / seg;
#pragma omp parallel for schedule(runtime)
    for (size_t i = 0; i < seg; i++) {
      size_t offset = length * i;
      size_t remain = size - offset;
      void *dst_ptr = reinterpret_cast<void *>(
          reinterpret_cast<uint8_t *>(buffer_->raw_mutable_data()) + offset);
      const void *src_ptr = reinterpret_cast<const void *>(
          reinterpret_cast<const uint8_t *>(src) + offset);
      memcpy(dst_ptr, src_ptr, std::min(length, remain));
    }
  }

  template <typename T>
  inline void Copy(const T *src, index_t length) {
    CONDITIONS(length == size(), "copy src and dst with different size.");
    CopyBytes(static_cast<const void *>(src), sizeof(T) * length);
  }

  inline void Copy(const Tensor &other) {
    dtype_ = other.dtype_;
    ResizeLike(other);
    MappingGuard map_other(&other);
    // CopyBytes(other.raw_data(), other.size() * SizeOfType());
    CopyBytesWithMultiCore(other.raw_data(), other.size() * SizeOfType());
  }

  inline size_t SizeOfType() const {
    size_t type_size = 0;
    DEEPVAN_RUN_WITH_TYPE_ENUM(dtype_, type_size = sizeof(T));
    return type_size;
  }

  inline BufferBase *UnderlyingBuffer() const { return buffer_; }

  inline void DebugPrint() const {
    using namespace numerical_chars; // NOLINT(build/namespaces)
    std::stringstream os;
    os << "Tensor " << name_ << " size: [";
    for (index_t i : shape_) {
      os << i << ", ";
    }
    os << "], content:\n";

    for (int i = 0; i < size(); ++i) {
      if (i != 0 && i % shape_.back() == 0) {
        os << "\n";
      }
      DEEPVAN_RUN_WITH_TYPE_ENUM(dtype_, (os << (this->data<T>()[i]) << ", "));
    }
    LOG(INFO) << os.str();
  }

  class MappingGuard {
  public:
    explicit MappingGuard(const Tensor *tensor) : tensor_(tensor) {
      if (tensor_ != nullptr) {
        CONDITIONS_NOTNULL(tensor_->buffer_);
        tensor_->buffer_->Map(&mapped_image_pitch_);
      }
    }

    MappingGuard(MappingGuard &&other) {
      tensor_ = other.tensor_;
      other.tensor_ = nullptr;
    }

    MappingGuard() {}

    ~MappingGuard() {
      if (tensor_ != nullptr)
        tensor_->buffer_->UnMap();
    }

    inline const std::vector<size_t> &mapped_image_pitch() const {
      return mapped_image_pitch_;
    }

    void map(const Tensor *tensor) {
      tensor_ = tensor;
      if (tensor_ != nullptr) {
        CONDITIONS_NOTNULL(tensor_->buffer_);
        tensor_->buffer_->Map(&mapped_image_pitch_);
      }
    }

  private:
    const Tensor *tensor_;
    std::vector<size_t> mapped_image_pitch_;

    DISABLE_COPY_AND_ASSIGN(MappingGuard);
  };

  inline bool is_weight() const { return is_weight_; }

  inline PruningType get_pruning_type() const { return pruning_type_; }

  inline void set_pruning_type(PruningType p) { pruning_type_ = p; }

  inline float scale() const { return scale_; }

  inline int32_t zero_point() const { return zero_point_; }

  inline float minval() const { return minval_; }

  inline float maxval() const { return maxval_; }

  inline void SetScale(float scale) { scale_ = scale; }

  inline void SetZeroPoint(int32_t zero_point) { zero_point_ = zero_point; }

  inline void SetIsWeight(bool is_weight) { is_weight_ = is_weight; }

  inline void SetMinVal(float minval) { minval_ = minval; }

  inline void SetMaxVal(float maxval) { maxval_ = maxval; }

private:
  Allocator *allocator_;
  DataType dtype_;
  // the shape of buffer(logical)
  std::vector<index_t> shape_;
  std::vector<index_t> shape_configured_;
  std::vector<size_t> image_shape_;
  // the shape of buffer(physical storage)
  std::vector<index_t> buffer_shape_;
  BufferBase *buffer_;
  BufferSlice buffer_slice_;
  bool is_buffer_owner_;
  bool unused_;
  std::string name_;
  bool is_weight_;
  float scale_;
  int32_t zero_point_;
  float minval_;
  float maxval_;
  DataFormat data_format_; // used for 4D input/output tensor
  // unused for CSR sparse data
  BufferSlice sparsed_row_;
  BufferSlice sparsed_col_;

  PruningType pruning_type_ = PruningType::DENSE;
  uint32_t weight_size_;
  uint32_t total_size_;
  std::vector<size_t> pattern_shape_; // oc * pruned_max_ic

  DISABLE_COPY_AND_ASSIGN(Tensor);
};
} // namespace deepvan

#endif // DEEPVAN_CORE_TENSOR_H_
