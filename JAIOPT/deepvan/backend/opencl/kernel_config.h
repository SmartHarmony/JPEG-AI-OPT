#include <vector>
#include <string>

namespace deepvan {
namespace opencl {

struct KernelConfigInfo {
  std::string kernel_name;
  std::vector<uint32_t> gws;
  std::vector<uint32_t> lws;
};

} // namespace opencl
} // namespace deepvan