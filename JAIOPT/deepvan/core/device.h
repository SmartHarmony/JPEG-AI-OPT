#ifndef DEEPVAN_CORE_DEVICE_H_
#define DEEPVAN_CORE_DEVICE_H_

#include <memory>

#include "deepvan/core/runtime/cpu/cpu_runtime.h"
#include "deepvan/core/allocator.h"

#ifdef OPENCL_SUPPORT
#include "deepvan/core/runtime/opencl/gpu_runtime.h"
#endif

namespace deepvan {
class ScratchBuffer;

class Device {
 public:
  virtual ~Device() {}

#ifdef OPENCL_SUPPORT
  virtual GPURuntime *gpu_runtime() = 0;
#endif  // OPENCL_SUPPORT

  virtual CPURuntime *cpu_runtime() = 0;

  virtual Allocator *allocator() = 0;
  virtual DeviceType device_type() const = 0;
  virtual ScratchBuffer *scratch_buffer() = 0;
};

class CPUDevice : public Device {
 public:
  CPUDevice(const CPUAffinityPolicySettings &cpu_affinity_settings);
  virtual ~CPUDevice();

#ifdef OPENCL_SUPPORT
  GPURuntime *gpu_runtime() override;
#endif
  CPURuntime *cpu_runtime() override;

  Allocator *allocator() override;
  DeviceType device_type() const override;
  ScratchBuffer *scratch_buffer() override;

 private:
  std::unique_ptr<CPURuntime> cpu_runtime_;
  std::unique_ptr<ScratchBuffer> scratch_buffer_;
};
}  // namespace deepvan
#endif  // DEEPVAN_CORE_DEVICE_H_
