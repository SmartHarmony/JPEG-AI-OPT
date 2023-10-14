#ifndef DEEPVAN_COMPAT_DARWIN_ENV_H_
#define DEEPVAN_COMPAT_DARWIN_ENV_H_

#include <string>
#include <vector>

#include "deepvan/compat/env.h"
#include "deepvan/compat/logger.h"
#include "deepvan/compat/posix/file_system.h"

namespace deepvan {
namespace compat {

class DarwinEnv : public Env {
 public:
  int64_t NowMicros() override;
  VanState GetCPUMaxFreq(std::vector<float> *max_freqs) override;
  FileSystem *GetFileSystem() override;
  LogWriter *GetLogWriter() override;
  std::vector<std::string> GetBackTraceUnsafe(int max_steps) override;

 private:
  PosixFileSystem posix_file_system_;
  LogWriter log_writer_;
};

}  // namespace compat
}  // namespace deepvan

#endif  // DEEPVAN_COMPAT_DARWIN_ENV_H_
