#include "deepvan/compat/linux_base/env.h"

#include <sys/time.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include "deepvan/compat/posix/file_system.h"
#include "deepvan/compat/posix/time.h"
#include "deepvan/utils/logging.h"

namespace deepvan {
namespace compat {


size_t LinuxBaseEnv::GetCPUCount() {
  size_t cpu_count = 0;
  std::string cpu_sys_conf = "/proc/cpuinfo";
  std::ifstream f(cpu_sys_conf);
  if (!f.is_open()) {
    LOG(ERROR) << "failed to open " << cpu_sys_conf;
    return -1;
  }
  std::string line;
  const std::string processor_key = "processor";
  while (std::getline(f, line)) {
    if (line.size() >= processor_key.size()
        && line.compare(0, processor_key.size(), processor_key) == 0) {
      ++cpu_count;
    }
  }
  if (f.bad()) {
    LOG(ERROR) << "failed to read " << cpu_sys_conf;
  }
  if (!f.eof()) {
    LOG(ERROR) << "failed to read end of " << cpu_sys_conf;
  }
  f.close();
  VLOG(1) << "CPU cores: " << cpu_count;
  return cpu_count;
}

int64_t LinuxBaseEnv::NowMicros() {
  return deepvan::compat::posix::NowMicros();
}

FileSystem *LinuxBaseEnv::GetFileSystem() {
  return &posix_file_system_;
}

VanState LinuxBaseEnv::GetCPUMaxFreq(std::vector<uint64_t> *max_freqs) {
  CONDITIONS_NOTNULL(max_freqs);
  int cpu_count = GetCPUCount();
  if (cpu_count < 0) {
    return VanState::RUNTIME_ERROR;
  }
  for (int cpu_id = 0; cpu_id < cpu_count; ++cpu_id) {
    std::string cpuinfo_max_freq_sys_conf = MakeString(
        "/sys/devices/system/cpu/cpu",
        cpu_id,
        "/cpufreq/cpuinfo_max_freq");
    std::ifstream f(cpuinfo_max_freq_sys_conf);
    if (!f.is_open()) {
      LOG(ERROR) << "failed to open " << cpuinfo_max_freq_sys_conf;
      return VanState::RUNTIME_ERROR;
    }
    std::string line;
    if (std::getline(f, line)) {
      uint64_t freq = strtoull(line.c_str(), nullptr, 10);
      max_freqs->push_back(freq);
    }
    if (f.bad()) {
      LOG(ERROR) << "failed to read " << cpuinfo_max_freq_sys_conf;
    }
    f.close();
  }

  VLOG(1) << "CPU freq: " << MakeString(*max_freqs);

  return VanState::SUCCEED;
}

VanState LinuxBaseEnv::GetCPUClusters(std::vector<CPUCluster> &clusters) {
  std::vector<uint64_t> cpu_max_freqs;
  RETURN_IF_ERROR(GetCPUMaxFreq(&cpu_max_freqs));
  if (cpu_max_freqs.empty()) {
    return VanState::RUNTIME_ERROR;
  }
  std::map<uint64_t, std::vector<size_t>> freq_id_map;
  for (size_t i = 0; i < cpu_max_freqs.size(); ++i) {
    freq_id_map[cpu_max_freqs[i]].push_back(i);
  }

  auto it = freq_id_map.begin();
  for (size_t i = 0; i < freq_id_map.size(); i++, it++) {
    clusters.push_back({i, it->second.size(), it->first, it->second});
    VLOG(1) << "Create a new CPU Cluster: " 
            <<  i << ", " 
            << it->second.size() << ", " 
            << it->first << ", " 
            << MakeString(it->second);
  }

  return VanState::SUCCEED;
}

}  // namespace compat
}  // namespace deepvan
