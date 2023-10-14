#ifndef DEEPVAN_CORE_BUFFER_H_
#define DEEPVAN_CORE_BUFFER_H_

#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>

#include "deepvan/core/allocator.h"
#include "deepvan/core/types.h"
#include "deepvan/utils/logging.h"
#include "deepvan/utils/macros.h"

namespace deepvan {
namespace core {
enum BufferType {
  BT_BUFFER,
  BT_IMAGE,
};
}  // namespace core

class BufferBase {
 public:
  BufferBase() : size_(0) {}
  explicit BufferBase(index_t size) : size_(size) {}
  virtual ~BufferBase() {}

  virtual core::BufferType buffer_type() const = 0;

  virtual void *buffer() = 0;

  virtual const void *raw_data() const = 0;

  virtual void *raw_mutable_data() = 0;

  virtual VanState Allocate(index_t nbytes) = 0;

  virtual VanState Allocate(const std::vector<size_t> &shape,
                              DataType data_type) = 0;

  virtual void *Map(index_t offset,
                    index_t length,
                    std::vector<size_t> *pitch) const = 0;

  virtual void UnMap(void *mapped_ptr) const = 0;

  virtual void Map(std::vector<size_t> *pitch) = 0;

  virtual void UnMap() = 0;

  virtual VanState Resize(index_t nbytes) = 0;

  virtual void Copy(void *src, index_t offset, index_t length) = 0;

  virtual bool OnHost() const = 0;

  virtual void Clear() = 0;

  virtual void Clear(index_t size) = 0;

  virtual const std::vector<size_t> shape() const = 0;

  virtual index_t offset() const { return 0; }

  template <typename T>
  const T *data() const {
    return reinterpret_cast<const T *>(raw_data());
  }

  template <typename T>
  T *mutable_data() {
    return reinterpret_cast<T *>(raw_mutable_data());
  }

  index_t size() const { return size_; }

 protected:
  index_t size_;
};

class Buffer : public BufferBase {
 public:
  explicit Buffer(Allocator *allocator)
      : BufferBase(0),
        allocator_(allocator),
        buf_(nullptr),
        mapped_buf_(nullptr),
        is_data_owner_(true) {}

  Buffer(Allocator *allocator, void *data, index_t size)
      : BufferBase(size),
        allocator_(allocator),
        buf_(data),
        mapped_buf_(nullptr),
        is_data_owner_(false) {}

  virtual ~Buffer() {
    if (mapped_buf_ != nullptr) {
      UnMap();
    }
    if (is_data_owner_ && buf_ != nullptr) {
      allocator_->Delete(buf_);
    }
  }

  core::BufferType buffer_type() const {
    return core::BufferType::BT_BUFFER;
  }

  void *buffer() {
    CONDITIONS_NOTNULL(buf_);
    return buf_;
  }

  const void *raw_data() const {
    if (OnHost()) {
      CONDITIONS_NOTNULL(buf_);
      return buf_;
    } else {
      CONDITIONS_NOTNULL(mapped_buf_);
      return mapped_buf_;
    }
  }

  void *raw_mutable_data() {
    if (OnHost()) {
      CONDITIONS_NOTNULL(buf_);
      return buf_;
    } else {
      CONDITIONS_NOTNULL(mapped_buf_);
      return mapped_buf_;
    }
  }

  VanState Allocate(index_t nbytes) {
    if (nbytes <= 0) {
      return VanState::SUCCEED;
    }
    CONDITIONS(is_data_owner_,
               "data is not owned by this buffer, cannot reallocate");
    if (mapped_buf_ != nullptr) {
      UnMap();
    }
    if (buf_ != nullptr) {
      allocator_->Delete(buf_);
    }
    size_ = nbytes;
    return allocator_->New(nbytes, &buf_);
  }

  VanState Allocate(const std::vector<size_t> &shape,
                      DataType data_type) {
    if (shape.empty()) return VanState::SUCCEED;
    index_t nbytes = std::accumulate(shape.begin(), shape.end(),
                                     1, std::multiplies<size_t>())
        * GetEnumTypeSize(data_type);
    return this->Allocate(nbytes);
  }

  void *Map(index_t offset, index_t length, std::vector<size_t> *pitch) const {
    CONDITIONS_NOTNULL(buf_);
    UNUSED_VARIABLE(pitch);
    return allocator_->Map(buf_, offset, length);
  }

  void UnMap(void *mapped_ptr) const {
    CONDITIONS_NOTNULL(buf_);
    CONDITIONS_NOTNULL(mapped_ptr);
    allocator_->Unmap(buf_, mapped_ptr);
  }

  void Map(std::vector<size_t> *pitch) {
    CONDITIONS(mapped_buf_ == nullptr, "buf has been already mapped");
    mapped_buf_ = Map(0, size_, pitch);
  }

  void UnMap() {
    UnMap(mapped_buf_);
    mapped_buf_ = nullptr;
  }

  VanState Resize(index_t nbytes) {
    CONDITIONS(is_data_owner_,
               "data is not owned by this buffer, cannot resize");
    if (nbytes != size_) {
      if (buf_ != nullptr) {
        allocator_->Delete(buf_);
      }
      size_ = nbytes;
      return allocator_->New(nbytes, &buf_);
    }
    return VanState::SUCCEED;
  }

  void Copy(void *src, index_t offset, index_t length) {
    CONDITIONS_NOTNULL(mapped_buf_);
    CONDITIONS(length <= size_, "out of buffer");
    memcpy(mapped_buf_, reinterpret_cast<char*>(src) + offset, length);
  }

  bool OnHost() const { return allocator_->OnHost(); }

  void Clear() {
    Clear(size_);
  }

  void Clear(index_t size) {
    memset(reinterpret_cast<char*>(raw_mutable_data()), 0, size);
  }

  const std::vector<size_t> shape() const {
    STUB;
    return {};
  }

 protected:
  Allocator *allocator_;
  void *buf_;
  void *mapped_buf_;
  bool is_data_owner_;

  DISABLE_COPY_AND_ASSIGN(Buffer);
};

class Image : public BufferBase {
 public:
  explicit Image(Allocator *allocator)
      : BufferBase(0),
        allocator_(allocator),
        buf_(nullptr),
        mapped_buf_(nullptr) {}

  virtual ~Image() {
    if (mapped_buf_ != nullptr) {
      UnMap();
    }
    if (buf_ != nullptr) {
      allocator_->DeleteImage(buf_);
    }
  }

  inline DataType dtype() const {
    CONDITIONS_NOTNULL(buf_);
    return data_type_;
  }

  core::BufferType buffer_type() const {
    return core::BufferType::BT_IMAGE;
  }

  void *buffer() {
    CONDITIONS_NOTNULL(buf_);
    return buf_;
  }

  const void *raw_data() const {
    CONDITIONS_NOTNULL(mapped_buf_);
    return mapped_buf_;
  }

  void *raw_mutable_data() {
    CONDITIONS_NOTNULL(mapped_buf_);
    return mapped_buf_;
  }

  VanState Allocate(index_t nbytes) {
    UNUSED_VARIABLE(nbytes);
    LOG(FATAL) << "Image should not call this allocate function";
    return VanState::SUCCEED;
  }

  VanState Allocate(const std::vector<size_t> &shape,
                      DataType data_type) {
    index_t size = std::accumulate(
        shape.begin(), shape.end(), 1, std::multiplies<index_t>()) *
        GetEnumTypeSize(data_type);
    if (mapped_buf_ != nullptr) {
      UnMap();
    }
    if (buf_ != nullptr) {
      allocator_->DeleteImage(buf_);
    }
    size_ = size;
    shape_ = shape;
    data_type_ = data_type;
    return allocator_->NewImage(shape, data_type, &buf_);
  }

  void *Map(index_t offset, index_t length, std::vector<size_t> *pitch) const {
    UNUSED_VARIABLE(offset);
    UNUSED_VARIABLE(length);
    UNUSED_VARIABLE(pitch);
    STUB;
    return nullptr;
  }

  void UnMap(void *mapped_ptr) const {
    CONDITIONS_NOTNULL(buf_);
    CONDITIONS_NOTNULL(mapped_ptr);
    allocator_->Unmap(buf_, mapped_ptr);
  }

  void Map(std::vector<size_t> *pitch) {
    CONDITIONS_NOTNULL(buf_);
    CONDITIONS(mapped_buf_ == nullptr, "buf has been already mapped");
    CONDITIONS_NOTNULL(pitch);
    mapped_buf_ = allocator_->MapImage(buf_, shape_, pitch);
  }

  void UnMap() {
    UnMap(mapped_buf_);
    mapped_buf_ = nullptr;
  }

  VanState Resize(index_t size) {
    UNUSED_VARIABLE(size);
    STUB;
    return VanState::SUCCEED;
  }

  void Copy(void *src, index_t offset, index_t length) {
    UNUSED_VARIABLE(src);
    UNUSED_VARIABLE(offset);
    UNUSED_VARIABLE(length);
    STUB;
  }

  bool OnHost() const { return allocator_->OnHost(); }

  void Clear() {
    STUB;
  }

  void Clear(index_t size) {
    UNUSED_VARIABLE(size);
    STUB;
  }

  const std::vector<size_t> shape() const {
    return shape_;
  }

 private:
  Allocator *allocator_;
  std::vector<size_t> shape_;
  DataType data_type_;
  void *buf_;
  void *mapped_buf_;

  DISABLE_COPY_AND_ASSIGN(Image);
};

class BufferSlice : public BufferBase {
 public:
  BufferSlice()
      : BufferBase(0), buffer_(nullptr), mapped_buf_(nullptr), offset_(0) {}
  BufferSlice(BufferBase *buffer, index_t offset, index_t length)
    : BufferBase(length),
      buffer_(buffer),
      mapped_buf_(nullptr),
      offset_(offset) {
    CONDITIONS(offset >= 0, "buffer slice offset should >= 0");
    CONDITIONS(offset + length <= buffer->size(),
               "buffer slice offset + length (",
               offset,
               " + ",
               length,
               ") should <= ",
               buffer->size());
  }
  BufferSlice(const BufferSlice &other)
      : BufferSlice(other.buffer_, other.offset_, other.size_) {}

  virtual ~BufferSlice() {
    if (buffer_ != nullptr && mapped_buf_ != nullptr) {
      UnMap();
    }
  }

  core::BufferType buffer_type() const {
    return core::BufferType::BT_BUFFER;
  }

  void *buffer() {
    CONDITIONS_NOTNULL(buffer_);
    return buffer_->buffer();
  }

  const void *raw_data() const {
    if (OnHost()) {
      CONDITIONS_NOTNULL(buffer_);
      return reinterpret_cast<const char*>(buffer_->raw_data()) + offset_;
    } else {
      CONDITIONS_NOTNULL(mapped_buf_);
      return mapped_buf_;
    }
  }

  void *raw_mutable_data() {
    if (OnHost()) {
      CONDITIONS_NOTNULL(buffer_);
      return reinterpret_cast<char*>(buffer_->raw_mutable_data()) + offset_;
    } else {
      CONDITIONS_NOTNULL(mapped_buf_);
      return mapped_buf_;
    }
  }

  VanState Allocate(index_t size) {
    UNUSED_VARIABLE(size);
    LOG(FATAL) << "BufferSlice should not call allocate function";
    return VanState::SUCCEED;
  }

  VanState Allocate(const std::vector<size_t> &shape,
                      DataType data_type) {
    UNUSED_VARIABLE(shape);
    UNUSED_VARIABLE(data_type);
    LOG(FATAL) << "BufferSlice should not call allocate function";
    return VanState::SUCCEED;
  }

  void *Map(index_t offset, index_t length, std::vector<size_t> *pitch) const {
    return buffer_->Map(offset_ + offset, length, pitch);
  }

  void UnMap(void *mapped_ptr) const {
    buffer_->UnMap(mapped_ptr);
  }

  void Map(std::vector<size_t> *pitch) {
    CONDITIONS_NOTNULL(buffer_);
    CONDITIONS(mapped_buf_ == nullptr, "mapped buf is not null");
    mapped_buf_ = buffer_->Map(offset_, size_, pitch);
  }

  void UnMap() {
    CONDITIONS_NOTNULL(mapped_buf_);
    buffer_->UnMap(mapped_buf_);
    mapped_buf_ = nullptr;
  }

  VanState Resize(index_t size) {
    CONDITIONS(size == size_, "resize buffer slice from ", size_,
      " to ", size, " is illegal");
    return VanState::SUCCEED;
  }

  void Copy(void *src, index_t offset, index_t length) {
    UNUSED_VARIABLE(src);
    UNUSED_VARIABLE(offset);
    UNUSED_VARIABLE(length);
    STUB;
  }

  index_t offset() const { return offset_; }

  bool OnHost() const { return buffer_->OnHost(); }

  void Clear() {
    Clear(size_);
  }

  void Clear(index_t size) {
    memset(raw_mutable_data(), 0, size);
  }

  const std::vector<size_t> shape() const {
    STUB;
    return {};
  }

 private:
  BufferBase *buffer_;
  void *mapped_buf_;
  index_t offset_;
};

class ScratchBuffer: public Buffer {
 public:
  explicit ScratchBuffer(Allocator *allocator)
    : Buffer(allocator),
      offset_(0) {}

  ScratchBuffer(Allocator *allocator, void *data, index_t size)
    : Buffer(allocator, data, size),
      offset_(0) {}

  virtual ~ScratchBuffer() {}

  VanState GrowSize(const index_t size) {
    if (offset_ + size > size_) {
      VLOG(1) << "Grow scratch size to: " << size 
              << ", offset: " << offset_;      
      CONDITIONS(offset_ == 0, "scratch is being used, cannot grow size");
      return Resize(size);
    }
    return VanState::SUCCEED;
  }

  // TODO @vgod [Check correctness] Grow size safty, which should:
  // 1. Allocate new memory to hold the previous data and new data
  // 2. Copy previous data into new memory
  // 3. Initialize offset
  VanState GrowSizeSafety(const index_t size) {
    if (offset_ + size > size_) {
      const index_t previous_data_size = offset_;
      char *tmp_data = (char *)malloc(previous_data_size);
      memcpy(tmp_data, buf_, previous_data_size);
      VanState state = Resize(previous_data_size + size);
      memcpy(buf_, tmp_data, previous_data_size);
      free(tmp_data);
      CONDITIONS(state == VanState::SUCCEED, "Grow size safety failed.");
    }
    return VanState::SUCCEED;
  }

  BufferSlice Scratch(index_t size) {
    CONDITIONS(offset_ + size <= size_,
               "scratch size not enough: ",
               offset_,
               " + ",
               size,
               " > ",
               size_);
    BufferSlice slice(this, offset_, size);
    offset_ += size;
    return slice;
  }

  void Rewind(index_t offset = 0) {
    offset_ = offset;
  }

  index_t offset() const {
    return offset_;
  }

 private:
  index_t offset_;
};
}  // namespace deepvan

#endif  // DEEPVAN_CORE_BUFFER_H_
