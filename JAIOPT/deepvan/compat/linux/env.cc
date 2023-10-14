#include "deepvan/compat/linux/env.h"

#include <execinfo.h>
#include <sys/time.h>

#include <cstddef>
#include <string>
#include <vector>

#include "deepvan/compat/env.h"
#include "deepvan/compat/posix/backtrace.h"
#include "deepvan/compat/posix/file_system.h"
#include "deepvan/compat/posix/time.h"

namespace deepvan {
namespace compat {

LogWriter *LinuxEnv::GetLogWriter() {
  return &log_writer_;
}

std::vector<std::string> LinuxEnv::GetBackTraceUnsafe(int max_steps) {
  return deepvan::compat::posix::GetBackTraceUnsafe(max_steps);
}

Env *Env::Default() {
  static LinuxEnv linux_env;
  return &linux_env;
}

}  // namespace compat
}  // namespace deepvan
