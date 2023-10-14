#ifndef DEEPVAN_COMPAT_ANDROID_ENV_H_
#define DEEPVAN_COMPAT_ANDROID_ENV_H_

#include <memory>
#include <string>
#include <vector>

#include "deepvan/compat/android/logger.h"
#include "deepvan/compat/env.h"
#include "deepvan/compat/linux_base/env.h"
#include "deepvan/compat/posix/file_system.h"

namespace deepvan {
namespace compat {

class AndroidEnv : public LinuxBaseEnv {
 public:
  VanState SchedSetAffinity(const std::vector<size_t> &cpu_ids) override;
  LogWriter *GetLogWriter() override;
  std::vector<std::string> GetBackTraceUnsafe(int max_steps) override;
  std::unique_ptr<MallocLogger> NewMallocLogger(std::ostringstream *oss, const std::string &name) override;

 private:
  AndroidLogWriter log_writer_;
};

}  // namespace compat
}  // namespace deepvan

#endif  // DEEPVAN_COMPAT_ANDROID_ENV_H_
