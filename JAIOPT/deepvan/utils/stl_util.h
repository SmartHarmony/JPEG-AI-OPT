#ifndef DEEPVAN_UTILS_STL_UTIL_H_
#define DEEPVAN_UTILS_STL_UTIL_H_

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace deepvan {
template <typename T>
std::vector<std::string> MapKeys(const std::map<std::string, T> &data) {
  std::vector<std::string> keys;
  for (auto &kv : data) {
    keys.push_back(kv.first);
  }
  return keys;
}
}  // namespace deepvan

#endif  // DEEPVAN_UTILS_STL_UTIL_H_
