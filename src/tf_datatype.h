#ifndef PATHFINDER_DRIVER_GENERATOR_TF_DATATYPE
#define PATHFINDER_DRIVER_GENERATOR_TF_DATATYPE

#include <optional>
#include <string>
#include <vector>

enum DataType {
  // tensorflow/core/framework/type.proto

  // flaoting
  DT_HALF,
  DT_BFLOAT16,
  DT_FLOAT,
  DT_DOUBLE,
  DT_FLOAT8_E4M3FN,
  DT_FLOAT8_E5M2,

  // integer
  DT_INT4,
  DT_UINT4,
  DT_INT8,
  DT_UINT8,
  DT_INT16,
  DT_UINT16,
  DT_INT32,
  DT_UINT32,
  DT_INT64,
  DT_UINT64,

  // complex
  DT_COMPLEX64,
  DT_COMPLEX128,

  // quantized
  DT_QINT8,
  DT_QUINT8,
  DT_QINT16,
  DT_QUINT16,
  DT_QINT32,

  DT_STRING,
  DT_BOOL,

  // How they should be handled?
  DT_RESOURCE,
  DT_VARIANT,
};

const std::vector<DataType>& all_datatypes();
std::string string_from_datatype(DataType datatype);
std::optional<DataType> datatype_from_string(const std::string& dt_string);
bool datatype_not_supported(DataType datatype);
bool is_integer_datatype(DataType datatype);
bool is_integer_datatype(const std::string& datatype_str);
bool is_integer_datatype(const std::vector<std::string>& datatype_strs);

#endif
