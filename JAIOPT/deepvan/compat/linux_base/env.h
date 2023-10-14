#ifndef DEEPVAN_COMPAT_LINUX_BASE_ENV_H_
#define DEEPVAN_COMPAT_LINUX_BASE_ENV_H_

#include <vector>

#include "deepvan/compat/env.h"
#include "deepvan/compat/posix/file_system.h"

namespace deepvan {
namespace compat {

class LinuxBaseEnv : public Env {
 public:
  int64_t NowMicros() override;
  size_t GetCPUCount() override;
  VanState GetCPUClusters(std::vector<CPUCluster> &clusters) override;
  VanState GetCPUMaxFreq(std::vector<uint64_t> *max_freqs) override;
  FileSystem *GetFileSystem() override;

 protected:
  PosixFileSystem posix_file_system_;
};

}  // namespace compat
}  // namespace deepvan

#endif  // DEEPVAN_COMPAT_LINUX_BASE_ENV_H_
