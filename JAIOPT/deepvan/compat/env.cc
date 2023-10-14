#include "deepvan/compat/env.h"

#include <sstream>

#include "deepvan/utils/memory.h"
#include "deepvan/export/deepvan.h"

namespace deepvan {

size_t GetCPUCount() {
  return compat::Env::Default()->GetCPUCount();
}

VanState GetCPUClusters(std::vector<CPUCluster> &clusters) {
  return compat::Env::Default()->GetCPUClusters(clusters);
}

namespace compat {

size_t Env::GetCPUCount() {
  return 0;
}

VanState Env::GetCPUMaxFreq(std::vector<uint64_t> *max_freqs) {
  return VanState::UNSUPPORTED;
}

VanState Env::SchedSetAffinity(const std::vector<size_t> &cpu_ids) {
  return VanState::UNSUPPORTED;
}

std::unique_ptr<MallocLogger> Env::NewMallocLogger(
      std::ostringstream *oss,
      const std::string &name) {
  return make_unique<MallocLogger>();
}

}  // namespace compat
}  // namespace deepvan
