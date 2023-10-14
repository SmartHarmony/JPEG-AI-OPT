#ifndef DEEPVAN_CORE_PREALLOCATED_POOLED_ALLOCATOR_H_
#define DEEPVAN_CORE_PREALLOCATED_POOLED_ALLOCATOR_H_

#include <memory>
#include <utility>
#include <unordered_map>

#include "deepvan/core/allocator.h"
#include "deepvan/core/buffer.h"

namespace deepvan {
class PreallocatedPooledAllocator {
 public:
  PreallocatedPooledAllocator() {}

  ~PreallocatedPooledAllocator() noexcept {}

  void SetBuffer(int mem_id, std::unique_ptr<BufferBase> &&buffer) {
    buffers_[mem_id] = std::move(buffer);
  }

  BufferBase *GetBuffer(int mem_id) {
    if (buffers_.find(mem_id) != buffers_.end()) {
      return buffers_[mem_id].get();
    } else {
      return nullptr;
    }
  }

  virtual bool HasBuffer(int mem_id) {
    return buffers_.find(mem_id) != buffers_.end();
  }

 private:
  std::unordered_map<int, std::unique_ptr<BufferBase>> buffers_;
};
}  // namespace deepvan

#endif  // DEEPVAN_CORE_PREALLOCATED_POOLED_ALLOCATOR_H_
