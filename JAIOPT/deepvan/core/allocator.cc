#include "deepvan/core/allocator.h"

#include <unistd.h>
#include <sys/mman.h>
#include <memory>

namespace deepvan {
Allocator *GetCPUAllocator() {
  static CPUAllocator allocator;
  return &allocator;
}

void AdviseFree(void *addr, size_t length) {
  int page_size = sysconf(_SC_PAGESIZE);
  void *addr_aligned =
      reinterpret_cast<void *>(
          (reinterpret_cast<uintptr_t>(addr) + page_size - 1)
              & (~(page_size - 1)));
  uintptr_t delta =
      reinterpret_cast<uintptr_t>(addr_aligned)
          - reinterpret_cast<uintptr_t>(addr);
  if (length >= delta + page_size) {
    size_t len_aligned = (length - delta) & (~(page_size - 1));
    int ret = madvise(addr_aligned, len_aligned, MADV_DONTNEED);
    if (ret != 0) {
      LOG(ERROR) << "Advise free failed: " << strerror(errno);
    }
  }
}
}  // namespace deepvan
