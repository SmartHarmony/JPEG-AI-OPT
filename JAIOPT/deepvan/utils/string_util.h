#ifndef DEEPVAN_UTILS_STRING_UTIL_H_
#define DEEPVAN_UTILS_STRING_UTIL_H_

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace deepvan {
namespace string_util {

inline void MakeStringInternal(std::stringstream & /*ss*/) {}

template <typename T>
inline void MakeStringInternal(std::stringstream &ss, const T &t) {
  ss << t;
}

template <typename T, typename... Args>
inline void MakeStringInternal(std::stringstream &ss,
                               const T &t,
                               const Args &... args) {
  MakeStringInternal(ss, t);
  MakeStringInternal(ss, args...);
}

class StringFormatter {
 public:
  static std::string Table(const std::string &title,
                           const std::vector<std::string> &header,
                           const std::vector<std::vector<std::string>> &data);
};

}  // namespace string_util

template <typename... Args>
std::string MakeString(const Args &... args) {
  std::stringstream ss;
  string_util::MakeStringInternal(ss, args...);
  return ss.str();
}

template <typename T>
std::string MakeListString(const T *args, size_t size) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < size; ++i) {
    ss << args[i];
    if (i < size - 1) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

template <typename T>
std::string MakeString(const std::vector<T> &args) {
  return MakeListString(args.data(), args.size());
}

// Specializations for already-a-string types.
template <>
inline std::string MakeString(const std::string &str) {
  return str;
}

inline std::string MakeString(const char *c_str) { return std::string(c_str); }

inline std::string ToLower(const std::string &src) {
  std::string dest(src);
  std::transform(src.begin(), src.end(), dest.begin(), ::tolower);
  return dest;
}

inline std::string ToUpper(const std::string &src) {
  std::string dest(src);
  std::transform(src.begin(), src.end(), dest.begin(), ::toupper);
  return dest;
}

std::string ObfuscateString(const std::string &src,
                            const std::string &lookup_table);

std::string ObfuscateString(const std::string &src);

std::string ObfuscateSymbol(const std::string &src);

#ifdef DEEPVAN_OBFUSCATE_LITERALS
#define DEEPVAN_OBFUSCATE_STRING(str) ObfuscateString(str)
#define DEEPVAN_OBFUSCATE_SYMBOL(str) ObfuscateSymbol(str)
#else
#define DEEPVAN_OBFUSCATE_STRING(str) (str)
#define DEEPVAN_OBFUSCATE_SYMBOL(str) (str)
#endif

std::vector<std::string> Split(const std::string &str, char delims);
}  // namespace deepvan

#ifdef __cplusplus
extern "C" {
#endif
// void *__memcpy_aarch64_simd(void *__restrict, const void *__restrict, size_t);
#ifdef __cplusplus
}
#endif

#endif  // DEEPVAN_UTILS_STRING_UTIL_H_
