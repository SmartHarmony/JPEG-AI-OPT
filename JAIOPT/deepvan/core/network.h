#ifndef DEEPVAN_CORE_NETWORK_H_
#define DEEPVAN_CORE_NETWORK_H_

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "deepvan/core/operator.h"

namespace deepvan {
class RunMetadata;
class NetworkController;
class MemoryOptimizer;

class NetworkBase {
public:
  NetworkBase() noexcept = default;
  virtual ~NetworkBase() = default;

  virtual VanState Init() = 0;

  virtual VanState Run(RunMetadata *run_metadata = nullptr) = 0;

protected:
  DISABLE_COPY_AND_ASSIGN(NetworkBase);
};

class SimpleNetwork : public NetworkBase {
public:
  SimpleNetwork(const OpRegistryBase *op_registry,
                const NetProto *net_def,
                NetworkController *ws,
                Device *target_device,
                MemoryOptimizer *mem_optimizer);

  VanState Init() override;

  VanState Run(RunMetadata *run_metadata = nullptr) override;

private:
  void CreateGPUNet(const OpRegistryBase *op_registry,
                    const NetProto *net_def,
                    MemoryOptimizer *mem_optimizer);

  void CreateCPUNet(const OpRegistryBase *op_registry,
                    const NetProto *net_def,
                    MemoryOptimizer *mem_optimizer);

  std::unique_ptr<Operation> CreateOperation(
      const OpRegistryBase *op_registry,
      OpConstructContext *construct_context,
      std::shared_ptr<OperatorProto> op_def,
      bool has_data_format,
      bool is_quantize_model = false);
  void CheckTensorAndOperationInfo();

protected:
  NetworkController *ws_;
  Device *target_device_;
  // CPU is base device.
  std::unique_ptr<Device> cpu_device_;
  std::vector<std::unique_ptr<Operation>> operators_;

  DISABLE_COPY_AND_ASSIGN(SimpleNetwork);
};
} // namespace deepvan

#endif // DEEPVAN_CORE_NETWORK_H_
