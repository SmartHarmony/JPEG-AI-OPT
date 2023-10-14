#ifndef DEEPVAN_COMPAT_ANDROID_MALLOC_LOGGER_H_
#define DEEPVAN_COMPAT_ANDROID_MALLOC_LOGGER_H_

#include <malloc.h>

#include <string>

#include "deepvan/compat/env.h"

namespace deepvan {
namespace compat {

class AndroidMallocLogger : public MallocLogger {
 public:
  explicit AndroidMallocLogger(std::ostringstream *oss,
                               const std::string &name);
  ~AndroidMallocLogger() override;

 private:
  std::ostringstream *oss_;
  const std::string name_;
  struct mallinfo prev_;
};

}  // namespace compat
}  // namespace deepvan


#endif  // DEEPVAN_COMPAT_ANDROID_MALLOC_LOGGER_H_
