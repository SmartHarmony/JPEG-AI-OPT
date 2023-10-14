#ifndef DEEPVAN_COMPAT_ANDROID_LOGGER_H_
#define DEEPVAN_COMPAT_ANDROID_LOGGER_H_

#include "deepvan/compat/logger.h"

namespace deepvan {
namespace compat {

class AndroidLogWriter : public LogWriter {
 protected:
  void WriteLogMessage(const char *fname,
                       const int line,
                       const LogLevel severity,
                       const char *message) override;
};

}  // namespace compat
}  // namespace deepvan

#endif  // DEEPVAN_COMPAT_ANDROID_LOGGER_H_
