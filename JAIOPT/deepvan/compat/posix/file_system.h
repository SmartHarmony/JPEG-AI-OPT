#ifndef DEEPVAN_COMPAT_POSIX_FILE_SYSTEM_H_
#define DEEPVAN_COMPAT_POSIX_FILE_SYSTEM_H_

#include <string>
#include <memory>

#include "deepvan/compat/file_system.h"

namespace deepvan {
namespace compat {

class PosixFileSystem : public FileSystem {
 public:
  PosixFileSystem() = default;
  ~PosixFileSystem() override = default;
  VanState NewReadOnlyMemoryRegionFromFile(const char *fname,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override;
};

}  // namespace compat
}  // namespace deepvan

#endif  // DEEPVAN_COMPAT_POSIX_FILE_SYSTEM_H_
