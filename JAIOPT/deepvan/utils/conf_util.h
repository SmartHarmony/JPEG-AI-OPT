#ifndef DEEPVAN_UTILS_CONF_UTIL_H_
#define DEEPVAN_UTILS_CONF_UTIL_H_

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace deepvan {
inline bool EnvConfEnabled(std::string env_name) {
  char *env = getenv(env_name.c_str());
  return !(!env || env[0] == 0 || env[0] == '0');
}
}  // namespace deepvan

#endif  // DEEPVAN_UTILS_CONF_UTIL_H_
