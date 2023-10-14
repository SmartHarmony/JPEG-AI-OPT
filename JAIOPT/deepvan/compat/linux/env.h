#ifndef DEEPVAN_COMPAT_LINUX_ENV_H_
#define DEEPVAN_COMPAT_LINUX_ENV_H_

#include <string>
#include <vector>

#include "deepvan/compat/linux_base/env.h"
#include "deepvan/compat/logger.h"

namespace deepvan {
namespace compat {

class LinuxEnv : public LinuxBaseEnv {
 public:
  LogWriter *GetLogWriter() override;
  std::vector<std::string> GetBackTraceUnsafe(int max_steps) override;

 private:
  LogWriter log_writer_;
};

}  // namespace compat
}  // namespace deepvan

#endif  // DEEPVAN_COMPAT_LINUX_ENV_H_
