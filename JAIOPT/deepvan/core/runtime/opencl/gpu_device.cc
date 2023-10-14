#include "deepvan/core/runtime/opencl/gpu_device.h"

#include "deepvan/core/buffer.h"

namespace deepvan {
GPUDevice::GPUDevice(std::shared_ptr<Tuner<uint32_t>> tuner,
                     std::shared_ptr<KVStorage> opencl_cache_storage,
                     const GPUPriorityHint priority,
                     const GPUPerfHint perf,
                     std::shared_ptr<KVStorage> opencl_binary_storage,
                     const CPUAffinityPolicySettings &cpu_affinity_settings) :
    CPUDevice(cpu_affinity_settings),
    runtime_(new OpenCLRuntime(opencl_cache_storage, priority, perf,
                               opencl_binary_storage, tuner)),
    allocator_(new OpenCLAllocator(runtime_.get())),
    scratch_buffer_(new ScratchBuffer(allocator_.get())),
    gpu_runtime_(new GPURuntime(runtime_.get())) {}

GPUDevice::

GPUDevice::~GPUDevice() = default;

GPURuntime* GPUDevice::gpu_runtime() {
  return gpu_runtime_.get();
}

Allocator *GPUDevice::allocator() {
  return allocator_.get();
}

DeviceType GPUDevice::device_type() const {
  return DeviceType::GPU;
}

ScratchBuffer *GPUDevice::scratch_buffer() {
  return scratch_buffer_.get();
}
}  // namespace deepvan
