#ifndef DEEPVAN_CORE_NETWORK_CONTROLLER_H_
#define DEEPVAN_CORE_NETWORK_CONTROLLER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "deepvan/core/device.h"
#include "deepvan/core/preallocated_pooled_allocator.h"
#include "deepvan/core/tensor.h"
#include "deepvan/export/deepvan.h"

namespace deepvan {
class MemoryOptimizer;

class NetworkController {
public:
  typedef std::map<std::string, std::unique_ptr<Tensor>> TensorMap;

  NetworkController();
  ~NetworkController() {}

  Tensor *CreateTensor(const std::string &name,
                       Allocator *alloc,
                       DataType type,
                       bool is_weight = false);

  inline bool HasTensor(const std::string &name) const {
    return tensor_map_.find(name) != tensor_map_.end();
  }

  const Tensor *GetTensor(const std::string &name) const;

  Tensor *GetTensor(const std::string &name);

  std::vector<std::string> Tensors() const;

  VanState LoadModelTensor(const NetProto &net_def,
                           Device *device,
                           const unsigned char *model_data,
                           const unsigned char *other_data = nullptr);

  VanState PreallocateOutputTensor(const NetProto &net_def,
                                   const MemoryOptimizer *mem_optimizer,
                                   Device *device);

  void RemoveUnusedBuffer();

  void RemoveAllBuffer();

  void RemoveAndReloadBuffer(const NetProto &net_def,
                             const unsigned char *model_data,
                             Allocator *alloc);

  void RemoveTensor(const std::string &name);

private:
  TensorMap tensor_map_;

  std::unique_ptr<BufferBase> tensor_buffer_;
  std::unique_ptr<BufferBase> other_buffer_;

  PreallocatedPooledAllocator preallocated_allocator_;

  DISABLE_COPY_AND_ASSIGN(NetworkController);
};
} // namespace deepvan

#endif // DEEPVAN_CORE_NETWORK_CONTROLLER_H_
