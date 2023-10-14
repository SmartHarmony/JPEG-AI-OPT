#include "deepvan/compat/posix/file_system.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <memory>
#include <string>

#include "deepvan/utils/memory.h"

namespace deepvan {
namespace compat {

namespace {
class PosixReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  PosixReadOnlyMemoryRegion() = delete;
  PosixReadOnlyMemoryRegion(const void* addr, uint64_t length)
    : addr_(addr), length_(length) {}
  ~PosixReadOnlyMemoryRegion() override {
    if (length_ > 0) {
      munmap(const_cast<void *>(addr_), length_);
    }
  };
  const void *data() const override { return addr_; };
  uint64_t length() const override { return length_; };

 private:
  const void *addr_;
  const uint64_t length_;
};
}  // namespace

VanState PosixFileSystem::NewReadOnlyMemoryRegionFromFile(
    const char *fname,
    std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  VanState s = VanState(VanState::SUCCEED);
  int fd = open(fname, O_RDONLY);
  if (fd < 0) {
    // TODO@vgod check errno
    s = VanState(VanState::RUNTIME_ERROR);
  } else {
    struct stat st;
    fstat(fd, &st);
    if (st.st_size > 0) {
      const void* address =
        mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
      if (address == MAP_FAILED) {
        // TODO@vgod check errno
        s = VanState(VanState::RUNTIME_ERROR);
      } else {
        *result = make_unique<PosixReadOnlyMemoryRegion>(address, st.st_size);
      }
      close(fd);
    } else {
      // Empty file: mmap returns EINVAL (since Linux 2.6.12) length was 0
      *result = make_unique<PosixReadOnlyMemoryRegion>(nullptr, 0);
    }
  }
  return s;
}

}  // namespace compat
}  // namespace deepvan
