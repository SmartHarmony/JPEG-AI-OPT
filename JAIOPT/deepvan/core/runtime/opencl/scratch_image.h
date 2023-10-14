#ifndef DEEPVAN_CORE_RUNTIME_OPENCL_SCRATCH_IMAGE_H_
#define DEEPVAN_CORE_RUNTIME_OPENCL_SCRATCH_IMAGE_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "deepvan/core/buffer.h"

namespace deepvan {
class ScratchImageManager {
 public:
  ScratchImageManager();
  ~ScratchImageManager();

  Image *Spawn(Allocator *allocator,
               const std::vector<size_t> &shape,
               const DataType dt,
               int *id);

  void Deactive(int id);

 private:
  std::unordered_map<int, std::unique_ptr<Image>> images_;
  std::vector<int> reference_count_;
};

class ScratchImage {
 public:
  explicit ScratchImage(ScratchImageManager *);
  ~ScratchImage();

  Image *Scratch(Allocator *allocator,
                 const std::vector<size_t> &shape,
                 const DataType dt);

 private:
  ScratchImageManager *manager_;
  int id_;
};
}  // namespace deepvan
#endif  // DEEPVAN_CORE_RUNTIME_OPENCL_SCRATCH_IMAGE_H_
