#include "deepvan/compat/darwin/env.h"

#include <execinfo.h>
#include <stdint.h>
#include <sys/sysctl.h>
#include <sys/time.h>
#include <sys/types.h>

#include <cstddef>
#include <string>
#include <vector>

#include "deepvan/compat/posix/backtrace.h"
#include "deepvan/compat/posix/file_system.h"
#include "deepvan/compat/posix/time.h"
#include "deepvan/utils/logging.h"

namespace deepvan {
namespace compat {

namespace {
const char kCpuFrequencyMax[] = "hw.cpufrequency_max";
}

int64_t DarwinEnv::NowMicros() {
  return deepvan::compat::posix::NowMicros();
}

// TODO(@vgod): this func is not accurate, darwin does not support
// acquiring CPU frequencies, we need to reconsider the CPU scheduling
// strategy.
VanState DarwinEnv::GetCPUMaxFreq(std::vector<float> *max_freqs) {
  CONDITIONS_NOTNULL(max_freqs);

  uint64_t freq = 0;
  size_t size = sizeof(freq);
  int ret = sysctlbyname(kCpuFrequencyMax, &freq, &size, NULL, 0);
  if (ret < 0) {
    LOG(ERROR) << "failed to get property: " << kCpuFrequencyMax;
    return VanState::RUNTIME_ERROR;
  }
  max_freqs->push_back(freq);

  return VanState::SUCCEED;
}

FileSystem *DarwinEnv::GetFileSystem() {
  return &posix_file_system_;
}

LogWriter *DarwinEnv::GetLogWriter() {
  return &log_writer_;
}

std::vector<std::string> DarwinEnv::GetBackTraceUnsafe(int max_steps) {
  return deepvan::compat::posix::GetBackTraceUnsafe(max_steps);
}

Env *Env::Default() {
  static DarwinEnv darwin_env;
  return &darwin_env;
}

}  // namespace compat
}  // namespace deepvan
