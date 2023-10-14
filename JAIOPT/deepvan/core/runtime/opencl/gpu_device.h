#ifndef DEEPVAN_CORE_RUNTIME_OPENCL_GPU_DEVICE_H_
#define DEEPVAN_CORE_RUNTIME_OPENCL_GPU_DEVICE_H_

#include <memory>

#include "deepvan/core/device_context.h"
#include "deepvan/core/device.h"
#include "deepvan/core/runtime/opencl/gpu_runtime.h"
#include "deepvan/core/runtime/opencl/opencl_allocator.h"

namespace deepvan {
class GPUDevice : public CPUDevice {
 public:
  GPUDevice(std::shared_ptr<Tuner<uint32_t>> tuner,
            std::shared_ptr<KVStorage> opencl_cache_storage = nullptr,
            const GPUPriorityHint priority = GPUPriorityHint::PRIORITY_LOW,
            const GPUPerfHint perf = GPUPerfHint::PERF_NORMAL,
            std::shared_ptr<KVStorage> opencl_binary_storage = nullptr,
            const CPUAffinityPolicySettings &affinity_policy_settings = CPUAffinityPolicySettings());

  ~GPUDevice();
  GPURuntime *gpu_runtime() override;
  Allocator *allocator() override;
  DeviceType device_type() const override;
  ScratchBuffer *scratch_buffer() override;
 private:
  std::unique_ptr<OpenCLRuntime> runtime_;
  std::unique_ptr<OpenCLAllocator> allocator_;
  std::unique_ptr<ScratchBuffer> scratch_buffer_;
  std::unique_ptr<GPURuntime> gpu_runtime_;
};
}  // namespace deepvan
#endif  // DEEPVAN_CORE_RUNTIME_OPENCL_GPU_DEVICE_H_
