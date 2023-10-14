#ifndef DEEPVAN_COMPAT_ENV_H_
#define DEEPVAN_COMPAT_ENV_H_

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "deepvan/export/deepvan.h"

namespace deepvan {
namespace compat {

class MallocLogger {
 public:
  MallocLogger() = default;
  virtual ~MallocLogger() = default;
};

class FileSystem;
class LogWriter;

class Env {
 public:
  virtual int64_t NowMicros() = 0;
  virtual size_t GetCPUCount();
  virtual VanState GetCPUClusters(std::vector<CPUCluster> &clusters);
  virtual VanState GetCPUMaxFreq(std::vector<uint64_t> *max_freqs);
  virtual VanState SchedSetAffinity(const std::vector<size_t> &cpu_ids);
  virtual FileSystem *GetFileSystem() = 0;
  virtual LogWriter *GetLogWriter() = 0;
  // Return the current backtrace, will allocate memory inside the call
  // which may fail
  virtual std::vector<std::string> GetBackTraceUnsafe(int max_steps) = 0;
  virtual std::unique_ptr<MallocLogger> NewMallocLogger(
      std::ostringstream *oss,
      const std::string &name);

  static Env *Default();
};

}  // namespace compat

inline int64_t NowMicros() {
  return compat::Env::Default()->NowMicros();
}

inline VanState GetCPUMaxFreq(std::vector<uint64_t> *max_freqs) {
  return compat::Env::Default()->GetCPUMaxFreq(max_freqs);
}

inline VanState SchedSetAffinity(const std::vector<size_t> &cpu_ids) {
  return compat::Env::Default()->SchedSetAffinity(cpu_ids);
}

inline compat::FileSystem *GetFileSystem() {
  return compat::Env::Default()->GetFileSystem();
}
}  // namespace deepvan

#endif  // DEEPVAN_COMPAT_ENV_H_
