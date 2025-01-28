#ifndef FUZZER_UTIL
#define FUZZER_UTIL

#include <cassert>
#include <limits>
#include <optional>
#include <vector>

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/public/session.h"

using namespace tensorflow;

namespace fuzzer_util {

tensorflow::Input tf_shape(size_t rank, int* dims);

tensorflow::Input tf_intvec(DataType dtype, size_t intvec_size,
                            std::vector<long> intvec_base);

tensorflow::Input tf_tensor(const ::tensorflow::Scope& scope, bool is_ref,
                            DataType dtype, size_t rank, int* dims);

tensorflow::Input tf_int_tensor(const ::tensorflow::Scope& scope,
                                DataType dtype, size_t rank, int* dims,
                                size_t int_array_size, int* int_array);

std::vector<DataType>& dtype_dict();
DataType get_dtype(size_t i);

std::vector<double>& double_dict();
double get_double(size_t i);

std::vector<float>& float_dict();
float get_float(size_t i);

bool is_too_big(size_t rank, std::vector<long> shape);

}  // namespace fuzzer_util

#endif
