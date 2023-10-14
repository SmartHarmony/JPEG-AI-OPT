#ifndef DEEPVAN_CORE_RUNTIME_OPENCL_OPENCL_ALLOCATOR_H_
#define DEEPVAN_CORE_RUNTIME_OPENCL_OPENCL_ALLOCATOR_H_

#include <memory>
#include <vector>

#include "deepvan/core/allocator.h"
#include "deepvan/core/runtime/opencl/opencl_runtime.h"

namespace deepvan {
class OpenCLAllocator : public Allocator {
 public:
  explicit OpenCLAllocator(OpenCLRuntime *opencl_runtime);

  ~OpenCLAllocator() override;

  VanState New(size_t nbytes, void **result) const override;

  /*
   * Use Image2D with RGBA (128-bit) format to represent the image.
   *
   * @ shape : [depth, ..., height, width ].
   */
  VanState NewImage(const std::vector<size_t> &image_shape,
                      const DataType dt,
                      void **result) const override;

  void Delete(void *buffer) const override;

  void DeleteImage(void *buffer) const override;

  void *Map(void *buffer, size_t offset, size_t nbytes) const override;

  void *MapImage(void *buffer,
                 const std::vector<size_t> &image_shape,
                 std::vector<size_t> *mapped_image_pitch) const override;

  void Unmap(void *buffer, void *mapped_ptr) const override;

  bool OnHost() const override;

 private:
  OpenCLRuntime *opencl_runtime_;
};
}  // namespace deepvan

#endif  // DEEPVAN_CORE_RUNTIME_OPENCL_OPENCL_ALLOCATOR_H_
