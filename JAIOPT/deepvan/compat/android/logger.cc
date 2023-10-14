#include "deepvan/compat/android/logger.h"

#include <android/log.h>

#include <iostream>

namespace deepvan {
namespace compat {

void AndroidLogWriter::WriteLogMessage(const char *fname,
                                       const int line,
                                       const LogLevel severity,
                                       const char *message) {
  int android_log_level;
  switch (severity) {
    case INFO:
      android_log_level = ANDROID_LOG_INFO;
      break;
    case WARNING:
      android_log_level = ANDROID_LOG_WARN;
      break;
    case ERROR:
      android_log_level = ANDROID_LOG_ERROR;
      break;
    case FATAL:
      android_log_level = ANDROID_LOG_FATAL;
      break;
    default:
      android_log_level = ANDROID_LOG_ERROR;
      break;
  }

  std::stringstream ss;
  const char *const partial_name = strrchr(fname, '/');
  ss << (partial_name != nullptr ? partial_name + 1 : fname) << ":" << line
     << " " << message;
  __android_log_write(android_log_level, "DEEPVAN", ss.str().c_str());

  // Also log to stderr (for standalone Android apps) and abort.
  LogWriter::WriteLogMessage(fname, line, severity, message);
}

}  // namespace compat
}  // namespace deepvan
