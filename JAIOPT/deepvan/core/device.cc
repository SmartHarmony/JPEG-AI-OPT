#include "deepvan/core/device.h"

#include "deepvan/core/buffer.h"
#include "deepvan/utils/memory.h"

namespace deepvan {
// TODO: This is reserved for the its use in network.cc. 
// It doesn't set up affinity policy. Although it seems 
// not required, since it the affinity policy has been 
// aready setup in target_device, the behavior is still unclear. 
// Need more investigation.
CPUDevice::CPUDevice(const CPUAffinityPolicySettings &cpu_affinity_settings)
    : cpu_runtime_(make_unique<CPURuntime>(cpu_affinity_settings)),
      scratch_buffer_(make_unique<ScratchBuffer>(GetCPUAllocator())) {}

CPUDevice::~CPUDevice() = default;

CPURuntime *CPUDevice::cpu_runtime() {
  return cpu_runtime_.get();
}

#ifdef OPENCL_SUPPORT
GPURuntime *CPUDevice::gpu_runtime() {
  LOG(FATAL) << "CPU device should not call GPU Runtime";
  return nullptr;
}
#endif

Allocator *CPUDevice::allocator() {
  return GetCPUAllocator();
}

DeviceType CPUDevice::device_type() const {
  return DeviceType::CPU;
}

ScratchBuffer *CPUDevice::scratch_buffer() {
  return scratch_buffer_.get();
}
}  // namespace deepvan
