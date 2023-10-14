#ifndef DEEPVAN_CORE_DEVICE_CONTEXT_H_
#define DEEPVAN_CORE_DEVICE_CONTEXT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "deepvan/core/kv_storage.h"
#include "deepvan/utils/tuner.h"

namespace deepvan {
class GPUContext {
 public:
  GPUContext(const std::string &storage_path = "",
             const std::vector<std::string> &opencl_binary_path = {},
             const std::string &opencl_parameter_path = "",
             const unsigned char *opencl_binary_ptr = nullptr,
             const size_t opencl_binary_size = 0,
             const unsigned char *opencl_parameter_ptr = nullptr,
             const size_t opencl_parameter_size = 0);
  ~GPUContext();

  std::shared_ptr<KVStorage> opencl_binary_storage();
  std::shared_ptr<KVStorage> opencl_cache_storage();
  std::shared_ptr<Tuner<uint32_t>> opencl_tuner();

 private:
  std::unique_ptr<KVStorageFactory> storage_factory_;
  std::shared_ptr<Tuner<uint32_t>> opencl_tuner_;
  std::shared_ptr<KVStorage> opencl_binary_storage_;
  std::shared_ptr<KVStorage> opencl_cache_storage_;
};
}  // namespace deepvan
#endif  // DEEPVAN_CORE_DEVICE_CONTEXT_H_
