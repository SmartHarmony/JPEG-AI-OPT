#ifndef DEEPVAN_COMPAT_POSIX_BACKTRACE_H_
#define DEEPVAN_COMPAT_POSIX_BACKTRACE_H_

#include <execinfo.h>

#include <string>
#include <vector>

namespace deepvan {
namespace compat {
namespace posix {

inline std::vector<std::string> GetBackTraceUnsafe(int max_steps) {
  std::vector<void *> buffer(max_steps, 0);
  int steps = backtrace(buffer.data(), max_steps);

  std::vector<std::string> bt;
  char **symbols = backtrace_symbols(buffer.data(), steps);
  if (symbols != nullptr) {
    for (int i = 0; i < steps; i++) {
      bt.push_back(symbols[i]);
    }
  }
  return bt;
}

}  // namespace posix
}  // namespace compat
}  // namespace deepvan

#endif  // DEEPVAN_COMPAT_POSIX_BACKTRACE_H_
