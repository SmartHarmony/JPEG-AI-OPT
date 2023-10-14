#include <sstream>

#include "deepvan/export/deepvan.h"

namespace deepvan {
class VanState::Impl {
 public:
  explicit Impl(const Code code): code_(code), information_("") {}
  Impl(const Code code, const std::string &informaton)
      : code_(code), information_(informaton) {}
  ~Impl() = default;

  void SetCode(const Code code) { code_ = code; }
  Code code() const { return code_; }
  void SetInformation(const std::string &info) { information_ = info; }
  std::string information() const {
    if (information_.empty()) {
      return CodeToString();
    } else {
      return CodeToString() + ": " + information_;
    }
  }

 private:
  std::string CodeToString() const {
    switch (code_) {
      case VanState::SUCCEED:
        return "Success";
      case VanState::INVALID_ARGS:
        return "Invalid Arguments";
      case VanState::OUT_OF_RESOURCES:
        return "Out of resources";
      case UNSUPPORTED:
        return "Unsupported";
      case RUNTIME_ERROR:
        return "Runtime error";
      default:
        std::ostringstream os;
        os << code_;
        return os.str();
    }
  }

 private:
  VanState::Code code_;
  std::string information_;
};

VanState::VanState()
    : impl_(new VanState::Impl(VanState::SUCCEED)) {}
VanState::VanState(const Code code) : impl_(new VanState::Impl(code)) {}
VanState::VanState(const Code code, const std::string &information)
    : impl_(new VanState::Impl(code, information)) {}
VanState::VanState(const VanState &other)
    : impl_(new VanState::Impl(other.code(), other.information())) {}
VanState::VanState(VanState &&other)
    : impl_(new VanState::Impl(other.code(), other.information())) {}
VanState::~VanState() = default;

VanState& VanState::operator=(const VanState &other) {
  impl_->SetCode(other.code());
  impl_->SetInformation(other.information());
  return *this;
}
VanState& VanState::operator=(const VanState &&other) {
  impl_->SetCode(other.code());
  impl_->SetInformation(other.information());
  return *this;
}

VanState::Code VanState::code() const {
  return impl_->code();
}

std::string VanState::information() const {
  return impl_->information();
}

bool VanState::operator==(const VanState &other) const {
  return other.code() == impl_->code();
}

bool VanState::operator!=(const VanState &other) const {
  return other.code() != impl_->code();
}
}  // namespace deepvan
