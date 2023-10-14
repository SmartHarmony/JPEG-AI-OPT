#ifndef DEEPVAN_CORE_OP_CONTEXT_H_
#define DEEPVAN_CORE_OP_CONTEXT_H_

#include "deepvan/core/device.h"
#include "deepvan/core/network_controller.h"
#include "deepvan/core/future.h"
#include "deepvan/proto/deepvan.pb.h"

namespace deepvan {
class OpContext {
 public:
  OpContext(NetworkController *ws, Device *device);
  ~OpContext();
  void set_device(Device *device);
  Device *device() const;
  NetworkController *workspace() const;

  void set_future(StatsFuture *future);
  StatsFuture *future() const;

  void set_model_type(ModelType);
  ModelType model_type() const;

 private:
  Device *device_;
  NetworkController *ws_;
  StatsFuture *future_;
  ModelType model_type_;
  // metadata
};
}  // namespace deepvan
#endif  // DEEPVAN_CORE_OP_CONTEXT_H_
