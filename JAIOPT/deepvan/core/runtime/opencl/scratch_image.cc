#include "deepvan/core/runtime/opencl/scratch_image.h"

#include <utility>
#include <vector>

#include "deepvan/utils/memory.h"

namespace deepvan {
ScratchImageManager::ScratchImageManager() = default;
ScratchImageManager::~ScratchImageManager() = default;

Image *ScratchImageManager::Spawn(
    Allocator *allocator,
    const std::vector<size_t> &shape,
    const DataType dt,
    int *id) {
  // TODO@vgod: not optimal memory reuse strategy
  int found_image_idx = -1;
  int image_count = static_cast<int>(reference_count_.size());
  for (int i = 0; i < image_count; ++i) {
    int count = reference_count_[i];
    if (count == 0 && images_.at(i)->dtype() == dt) {
      auto image_shape = images_.at(i)->shape();
      if (image_shape[0] >= shape[0] && image_shape[1] >= shape[1]) {
        found_image_idx = i;
        break;
      }
    }
  }
  // if not found
  if (found_image_idx == -1) {
    reference_count_.push_back(0);
    images_[image_count] = make_unique<Image>(allocator);
    if (images_.at(image_count)->Allocate(shape, dt) !=
        VanState::SUCCEED) {
      return nullptr;
    }
    found_image_idx = image_count;
    VLOG(2) << "Spawn image " << found_image_idx << ": " << MakeString(shape)
            << "<" << dt << ">";
  }
  reference_count_[found_image_idx] += 1;
  *id = found_image_idx;
  return images_.at(found_image_idx).get();
}

void ScratchImageManager::Deactive(int id) {
  CONDITIONS(reference_count_.size() > static_cast<size_t>(id)
                 && reference_count_[id] > 0,
             "Image id ", id, " exceed the vector size ",
             reference_count_.size());
  reference_count_[id] -= 1;
}

ScratchImage::ScratchImage(deepvan::ScratchImageManager *manager)
    : manager_(manager), id_(-1) {}

ScratchImage::~ScratchImage() {
  if (id_ >= 0) {
    manager_->Deactive(id_);
  }
}

Image* ScratchImage::Scratch(Allocator *allocator,
                             const std::vector<size_t> &shape,
                             const deepvan::DataType dt) {
  return manager_->Spawn(allocator, shape, dt, &id_);
}
}  // namespace deepvan
