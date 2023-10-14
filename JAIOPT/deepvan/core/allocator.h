#ifndef DEEPVAN_CORE_ALLOCATOR_H_
#define DEEPVAN_CORE_ALLOCATOR_H_

#include <cstdlib>
#include <map>
#include <limits>
#include <vector>
#include <cstring>

#include "deepvan/utils/macros.h"
#include "deepvan/core/types.h"
#include "deepvan/export/deepvan.h"
#include "deepvan/utils/logging.h"

namespace deepvan {
#if defined(__hexagon__)
constexpr size_t kDeepvanAlignment = 128;
#elif defined(__ANDROID__)
// arm cache line
constexpr size_t kDeepvanAlignment = 64;
#else
// 32 bytes = 256 bits (AVX512)
constexpr size_t kDeepvanAlignment = 32;
#endif

inline index_t PadAlignSize(index_t size) {
  return (size + kDeepvanAlignment - 1) & (~(kDeepvanAlignment - 1));
}

class Allocator {
 public:
  Allocator() {}
  virtual ~Allocator() noexcept {}
  virtual VanState New(size_t nbytes, void **result) const = 0;
  virtual VanState NewImage(const std::vector<size_t> &image_shape,
                              const DataType dt,
                              void **result) const = 0;
  virtual void Delete(void *data) const = 0;
  virtual void DeleteImage(void *data) const = 0;
  virtual void *Map(void *buffer, size_t offset, size_t nbytes) const = 0;
  virtual void *MapImage(void *buffer,
                         const std::vector<size_t> &image_shape,
                         std::vector<size_t> *mapped_image_pitch) const = 0;
  virtual void Unmap(void *buffer, void *mapper_ptr) const = 0;
  virtual bool OnHost() const = 0;
};

class CPUAllocator : public Allocator {
 public:
  ~CPUAllocator() override {}
  VanState New(size_t nbytes, void **result) const override {
    VLOG(3) << "Allocate CPU buffer: " << nbytes;
    if (nbytes == 0) {
      return VanState::SUCCEED;
    }

    void *data = nullptr;
#if defined(__ANDROID__) || defined(__hexagon__)
    data = memalign(kDeepvanAlignment, nbytes);
    if (data == NULL) {
      LOG(WARNING) << "Allocate CPU Buffer with "
                   << nbytes << " bytes failed because of"
                   << strerror(errno);
      *result = nullptr;
      return VanState::OUT_OF_RESOURCES;
    }
#else
    int ret = posix_memalign(&data, kDeepvanAlignment, nbytes);
    if (ret != 0) {
      LOG(WARNING) << "Allocate CPU Buffer with "
                   << nbytes << " bytes failed because of"
                   << strerror(errno);
      if (data != NULL) {
        free(data);
      }
      *result = nullptr;
      return VanState::OUT_OF_RESOURCES;
    }
#endif
    // TODO@vgod This should be avoided sometimes
    memset(data, 0, nbytes);
    *result = data;
    return VanState::SUCCEED;
  }

  VanState NewImage(const std::vector<size_t> &shape,
                      const DataType dt,
                      void **result) const override {
    UNUSED_VARIABLE(shape);
    UNUSED_VARIABLE(dt);
    UNUSED_VARIABLE(result);
    LOG(FATAL) << "Allocate CPU image";
    return VanState::SUCCEED;
  }

  void Delete(void *data) const override {
    CONDITIONS_NOTNULL(data);
    VLOG(3) << "Free CPU buffer";
    free(data);
  }
  void DeleteImage(void *data) const override {
    LOG(FATAL) << "Free CPU image";
    free(data);
  };
  void *Map(void *buffer, size_t offset, size_t nbytes) const override {
    UNUSED_VARIABLE(nbytes);
    return reinterpret_cast<char*>(buffer) + offset;
  }
  void *MapImage(void *buffer,
                 const std::vector<size_t> &image_shape,
                 std::vector<size_t> *mapped_image_pitch) const override {
    UNUSED_VARIABLE(image_shape);
    UNUSED_VARIABLE(mapped_image_pitch);
    return buffer;
  }
  void Unmap(void *buffer, void *mapper_ptr) const override {
    UNUSED_VARIABLE(buffer);
    UNUSED_VARIABLE(mapper_ptr);
  }
  bool OnHost() const override { return true; }
};

// Global CPU allocator used for CPU/GPU/DSP
Allocator *GetCPUAllocator();

void AdviseFree(void *addr, size_t length);
}  // namespace deepvan

#endif  // DEEPVAN_CORE_ALLOCATOR_H_
