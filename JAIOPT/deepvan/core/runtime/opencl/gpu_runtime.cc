#include "deepvan/core/runtime/opencl/gpu_runtime.h"

#include "deepvan/core/runtime/opencl/scratch_image.h"

namespace deepvan {
GPURuntime::GPURuntime(deepvan::OpenCLRuntime *runtime)
    : runtime_(runtime),
      scratch_image_manager_(new ScratchImageManager),
      mem_type_(MemoryType::GPU_IMAGE) {}

GPURuntime::~GPURuntime() = default;

OpenCLRuntime* GPURuntime::opencl_runtime() {
  return runtime_;
}

ScratchImageManager* GPURuntime::scratch_image_manager() const {
  return scratch_image_manager_.get();
}

bool GPURuntime::UseImageMemory() {
  return this->mem_type_ == MemoryType::GPU_IMAGE;
}

void GPURuntime::set_mem_type(MemoryType type) {
  this->mem_type_ = type;
}

}  // namespace deepvan
