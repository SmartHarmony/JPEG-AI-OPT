#ifndef DEEPVAN_COMPAT_FILE_SYSTEM_H_
#define DEEPVAN_COMPAT_FILE_SYSTEM_H_

#include <string>
#include <memory>

#include "deepvan/export/deepvan.h"

namespace deepvan {
namespace compat {

class ReadOnlyMemoryRegion {
 public:
  ReadOnlyMemoryRegion() = default;
  virtual ~ReadOnlyMemoryRegion() = default;
  virtual const void *data() const = 0;
  virtual uint64_t length() const = 0;
};

class ReadOnlyBufferMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  ReadOnlyBufferMemoryRegion() : data_(nullptr), length_(0) {}
  ReadOnlyBufferMemoryRegion(const void *data, uint64_t length) :
    data_(data), length_(length) {}
  const void *data() const override { return data_; }
  uint64_t length() const override { return length_; }

 private:
  const void *data_;
  uint64_t length_;
};

class FileSystem {
 public:
  FileSystem() = default;
  virtual ~FileSystem() = default;
  virtual VanState NewReadOnlyMemoryRegionFromFile(const char *fname,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) = 0;
};

}  // namespace compat
}  // namespace deepvan

#endif  // DEEPVAN_COMPAT_FILE_SYSTEM_H_
