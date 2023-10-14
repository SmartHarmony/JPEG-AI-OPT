#ifndef DEEPVAN_BACKEND_COMMON_ACTIVATION_TYPE_H_
#define DEEPVAN_BACKEND_COMMON_ACTIVATION_TYPE_H_

#include "deepvan/utils/logging.h"
#include <string>

namespace deepvan {

enum ActivationType {
  NOOP = 0,
  RELU = 1,
  RELUX = 2,
  PRELU = 3,
  TANH = 4,
  SIGMOID = 5,
  LEAKYRELU = 6,
  ROUND = 7,
  HARDSIGMOID = 8,
  SWISH = 9,
  COS = 10
};

inline ActivationType StringToActivationType(const std::string type) {
  if (ToUpper(type) == "RELU") {
    return ActivationType::RELU;
  } else if (ToUpper(type) == "RELUX") {
    return ActivationType::RELUX;
  } else if (ToUpper(type) == "PRELU") {
    return ActivationType::PRELU;
  } else if (ToUpper(type) == "TANH") {
    return ActivationType::TANH;
  } else if (ToUpper(type) == "SIGMOID") {
    return ActivationType::SIGMOID;
  } else if (ToUpper(type) == "NOOP") {
    return ActivationType::NOOP;
  } else if (ToUpper(type) == "LEAKYRELU") {
    return ActivationType ::LEAKYRELU;
  } else if (type == "ROUND") {
    return ActivationType ::ROUND;
  } else if (type == "HARDSIGMOID") {
    return ActivationType ::HARDSIGMOID;
  } else if (ToUpper(type) == "SWISH") {
    return ActivationType ::SWISH;
  } else if (ToUpper(type) == "COS") {
    return ActivationType ::COS;
  } else {
    LOG(FATAL) << "Unknown activation type: " << type;
  }
  return ActivationType::NOOP;
}

} // namespace deepvan

#endif // DEEPVAN_BACKEND_COMMON_ACTIVATION_TYPE_H_
