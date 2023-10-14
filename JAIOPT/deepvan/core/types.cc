#include <cstdint>
#include <map>

#include "deepvan/core/types.h"
#include "deepvan/utils/logging.h"

namespace deepvan {
bool DataTypeCanUseMemcpy(DataType dt) {
  switch (dt) {
    case DT_FLOAT:
    case DT_UINT8:
    case DT_INT32:
      return true;
    default:
      return false;
  }
}

std::string DataTypeToString(const DataType dt) {
  static std::map<DataType, std::string> dtype_string_map = {
      {DT_FLOAT, "DT_FLOAT"},
      {DT_HALF, "DT_HALF"},
      {DT_UINT8, "DT_UINT8"},
      {DT_INT32, "DT_INT32"}};
  CONDITIONS(dt != DT_INVALID, "Not support Invalid data type");
  return dtype_string_map[dt];
}

size_t GetEnumTypeSize(const DataType dt) {
  switch (dt) {
    case DT_FLOAT:
      return sizeof(float);
    case DT_HALF:
      return sizeof(half);
    case DT_UINT8:
      return sizeof(uint8_t);
    case DT_INT32:
      return sizeof(int32_t);
    default:
      LOG(FATAL) << "Unsupported data type: " << dt;
      return 0;
  }
}
}  // namespace deepvan
