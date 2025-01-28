#include "tf_datatype.h"

#include <cassert>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "utils.h"

const std::vector<DataType>& all_datatypes() {
  static const std::vector<DataType> all_datatypes_ = {
      DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE,
      // DT_FLOAT8_E4M3FN,
      // DT_FLOAT8_E5M2,

      DT_INT4, DT_UINT4, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
      DT_UINT32, DT_INT64, DT_UINT64,

      DT_COMPLEX64, DT_COMPLEX128,

      DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32,

      DT_STRING, DT_BOOL,

      // DT_RESOURCE,
      // DT_VARIANT,
  };
  return all_datatypes_;
}

std::string string_from_datatype(DataType datatype) {
  if (datatype == DT_HALF) {
    return "DT_HALF";
  } else if (datatype == DT_BFLOAT16) {
    return "DT_BFLOAT16";
  } else if (datatype == DT_FLOAT) {
    return "DT_FLOAT";
  } else if (datatype == DT_DOUBLE) {
    return "DT_DOUBLE";
  } else if (datatype == DT_FLOAT8_E4M3FN) {
    return "DT_FLOAT8_E4M3FN";
  } else if (datatype == DT_FLOAT8_E5M2) {
    return "DT_FLOAT8_E5M2";
  } else if (datatype == DT_INT4) {
    return "DT_INT4";
  } else if (datatype == DT_UINT4) {
    return "DT_UINT4";
  } else if (datatype == DT_INT8) {
    return "DT_INT8";
  } else if (datatype == DT_UINT8) {
    return "DT_UINT8";
  } else if (datatype == DT_INT16) {
    return "DT_INT16";
  } else if (datatype == DT_UINT16) {
    return "DT_UINT16";
  } else if (datatype == DT_INT32) {
    return "DT_INT32";
  } else if (datatype == DT_UINT32) {
    return "DT_UINT32";
  } else if (datatype == DT_INT64) {
    return "DT_INT64";
  } else if (datatype == DT_UINT64) {
    return "DT_UINT64";
  } else if (datatype == DT_COMPLEX64) {
    return "DT_COMPLEX64";
  } else if (datatype == DT_COMPLEX128) {
    return "DT_COMPLEX128";
  } else if (datatype == DT_QINT8) {
    return "DT_QINT8";
  } else if (datatype == DT_QUINT8) {
    return "DT_QUINT8";
  } else if (datatype == DT_QINT16) {
    return "DT_QINT16";
  } else if (datatype == DT_QUINT16) {
    return "DT_QUINT16";
  } else if (datatype == DT_QINT32) {
    return "DT_QINT32";
  } else if (datatype == DT_STRING) {
    return "DT_STRING";
  } else if (datatype == DT_BOOL) {
    return "DT_BOOL";
  } else if (datatype == DT_RESOURCE) {
    return "DT_RESOURCE";
  } else if (datatype == DT_VARIANT) {
    return "DT_VARIANT";
  }
  UNREACHABLE;
}

std::optional<DataType> datatype_from_string(const std::string& dt_string) {
  if (dt_string == "half" || dt_string == "DT_HALF") {
    return DT_HALF;
  } else if (dt_string == "bfloat16" || dt_string == "DT_BFLOAT16") {
    return DT_BFLOAT16;
  } else if (dt_string == "float" || dt_string == "DT_FLOAT") {
    return DT_FLOAT;
  } else if (dt_string == "double" || dt_string == "DT_DOUBLE") {
    return DT_DOUBLE;
  } else if (dt_string == "DT_FLOAT8_E4M3FN") {
    return DT_FLOAT8_E4M3FN;
  } else if (dt_string == "DT_FLOAT8_E5M2") {
    return DT_FLOAT8_E5M2;
  } else if (dt_string == "int4" || dt_string == "DT_INT4") {
    return DT_INT4;
  } else if (dt_string == "uint4" || dt_string == "DT_UINT4") {
    return DT_UINT4;
  } else if (dt_string == "int8" || dt_string == "DT_INT8") {
    return DT_INT8;
  } else if (dt_string == "uint8" || dt_string == "DT_UINT8") {
    return DT_UINT8;
  } else if (dt_string == "int16" || dt_string == "DT_INT16") {
    return DT_INT16;
  } else if (dt_string == "uint16" || dt_string == "DT_UINT16") {
    return DT_UINT16;
  } else if (dt_string == "int" || dt_string == "int32" ||
             dt_string == "DT_INT32") {
    return DT_INT32;
  } else if (dt_string == "uint32" || dt_string == "DT_UINT32") {
    return DT_UINT32;
  } else if (dt_string == "int64" || dt_string == "DT_INT64") {
    return DT_INT64;
  } else if (dt_string == "uint64" || dt_string == "DT_UINT64") {
    return DT_UINT64;
  } else if (dt_string == "complex64" || dt_string == "DT_COMPLEX64") {
    return DT_COMPLEX64;
  } else if (dt_string == "complex128" || dt_string == "DT_COMPLEX128") {
    return DT_COMPLEX128;
  } else if (dt_string == "DT_QINT8") {
    return DT_QINT8;
  } else if (dt_string == "DT_QUINT8") {
    return DT_QUINT8;
  } else if (dt_string == "DT_QINT16") {
    return DT_QINT16;
  } else if (dt_string == "DT_QUINT16") {
    return DT_QUINT16;
  } else if (dt_string == "DT_QINT32") {
    return DT_QINT32;
  } else if (dt_string == "string" || dt_string == "DT_STRING") {
    return DT_STRING;
  } else if (dt_string == "bool" || dt_string == "DT_BOOL") {
    return DT_BOOL;
  } else if (dt_string == "resource" || dt_string == "DT_RESOURCE") {
    return DT_RESOURCE;
  } else if (dt_string == "variant" || dt_string == "DT_VARIANT") {
    return DT_VARIANT;
  }

  return std::nullopt;
}

bool datatype_not_supported(DataType datatype) {
  if (datatype == DT_RESOURCE || datatype == DT_VARIANT) {
    return true;
  }
  return false;
}

bool is_integer_datatype(DataType datatype) {
  if (datatype == DT_INT4 || datatype == DT_UINT4 || datatype == DT_INT8 ||
      datatype == DT_UINT8 || datatype == DT_INT16 || datatype == DT_UINT16 ||
      datatype == DT_INT32 || datatype == DT_UINT32 || datatype == DT_INT64 ||
      datatype == DT_UINT64)
    return true;
  return false;
}

bool is_integer_datatype(const std::string& datatype_str) {
  auto datatype = datatype_from_string(datatype_str);
  assert(datatype.has_value());
  return is_integer_datatype(datatype.value());
}

bool is_integer_datatype(const std::vector<std::string>& datatype_strs) {
  for (auto& datatype_str : datatype_strs)
    if (!is_integer_datatype(datatype_str)) return false;
  return true;
}
