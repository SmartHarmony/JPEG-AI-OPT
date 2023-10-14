#ifndef DEEPVAN_CORE_TYPES_H_
#define DEEPVAN_CORE_TYPES_H_

#include <cstdint>
#include <string>

#include "deepvan/proto/deepvan.pb.h"
#include "include/half.hpp"

namespace deepvan {

enum FrameworkType {
  TENSORFLOW = 0,
  CAFFE = 1,
  ONNX = 2,
};

typedef int64_t index_t;

using half = half_float::half;

bool DataTypeCanUseMemcpy(DataType dt);

size_t GetEnumTypeSize(const DataType dt);

std::string DataTypeToString(const DataType dt);

template <class T>
struct DataTypeToEnum;

template <DataType VALUE>
struct EnumToDataType;

#define DEEPVAN_MAPPING_DATA_TYPE_AND_ENUM(DATA_TYPE, ENUM_VALUE)  \
  template <>                                                   \
  struct DataTypeToEnum<DATA_TYPE> {                            \
    static DataType v() { return ENUM_VALUE; }                  \
    static constexpr DataType value = ENUM_VALUE;               \
  };                                                            \
  template <>                                                   \
  struct EnumToDataType<ENUM_VALUE> {                           \
    typedef DATA_TYPE Type;                                     \
  };

DEEPVAN_MAPPING_DATA_TYPE_AND_ENUM(half, DT_HALF);
DEEPVAN_MAPPING_DATA_TYPE_AND_ENUM(float, DT_FLOAT);
DEEPVAN_MAPPING_DATA_TYPE_AND_ENUM(uint8_t, DT_UINT8);
DEEPVAN_MAPPING_DATA_TYPE_AND_ENUM(int32_t, DT_INT32);
}  // namespace deepvan

#endif  // DEEPVAN_CORE_TYPES_H_
