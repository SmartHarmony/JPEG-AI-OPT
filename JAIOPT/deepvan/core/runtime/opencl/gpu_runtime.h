#ifndef DEEPVAN_CORE_RUNTIME_OPENCL_GPU_RUNTIME_H_
#define DEEPVAN_CORE_RUNTIME_OPENCL_GPU_RUNTIME_H_

#include <memory>

#include "deepvan/proto/deepvan.pb.h"

namespace deepvan {
class OpenCLRuntime;
class ScratchImageManager;

class GPURuntime {
 public:
  explicit GPURuntime(OpenCLRuntime *runtime);
  ~GPURuntime();
  OpenCLRuntime *opencl_runtime();
  ScratchImageManager *scratch_image_manager() const;

  // TODO@vgod: remove this function in the future, make decision at runtime.
  bool UseImageMemory();
  void set_mem_type(MemoryType type);

 private:
  OpenCLRuntime *runtime_;
  std::unique_ptr<ScratchImageManager> scratch_image_manager_;
  MemoryType mem_type_;
};
}  // namespace deepvan
#endif  // DEEPVAN_CORE_RUNTIME_OPENCL_GPU_RUNTIME_H_
