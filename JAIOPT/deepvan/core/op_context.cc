#include "deepvan/core/op_context.h"

namespace deepvan {
OpContext::OpContext(NetworkController *ws, Device *device)
    : device_(device), ws_(ws), future_(nullptr) {}

OpContext::~OpContext() = default;

void OpContext::set_device(Device *device) {
  device_ = device;
}

Device* OpContext::device() const {
  return device_;
}

NetworkController* OpContext::workspace() const {
  return ws_;
}

void OpContext::set_future(StatsFuture *future) {
  future_ = future;
}

StatsFuture *OpContext::future() const {
  return future_;
}

void OpContext::set_model_type(ModelType model_type) {
  model_type_ = model_type;
}

ModelType OpContext::model_type() const{
  return model_type_;
}
}  // namespace deepvan
