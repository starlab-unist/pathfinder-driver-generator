#include "fuzzer_util.h"

namespace fuzzer_util {

static const size_t RANK_MAX = 5;
static const size_t ARRAY_SIZE_MAX = 6;

tensorflow::Input tf_shape(size_t rank, int* dims) {
  assert(rank <= RANK_MAX);
  switch (rank) {
    case 1:
      return {dims[0]};
    case 2:
      return {dims[0], dims[1]};
    case 3:
      return {dims[0], dims[1], dims[2]};
    case 4:
      return {dims[0], dims[1], dims[2], dims[3]};
    case 5:
      return {dims[0], dims[1], dims[2], dims[3], dims[4]};
  }
  return Tensor(DT_INT32, TensorShape({0}));
}

tensorflow::Input tf_tensor(const ::tensorflow::Scope& scope, bool is_ref,
                            DataType dtype, size_t rank, int* dims) {
  assert(rank <= RANK_MAX);

  if (is_ref) {
    PartialTensorShape shape;
    TF_CHECK_OK(PartialTensorShape::MakePartialShape(dims, rank, &shape));
    return ops::Variable(scope, shape, dtype);
  }

  if (DataTypeIsFloating(dtype)) {
    auto base = ops::RandomNormal(scope, tf_shape(rank, dims), DT_FLOAT);
    return ops::Cast(scope, base, dtype);
  }

  if (DataTypeIsInteger(dtype)) {
    int maxval;
    switch (dtype) {
      case DT_INT4:
        maxval = 7;
        break;
      case DT_UINT4:
        maxval = 15;
        break;
      default:
        maxval = 64;
        break;
    }
    auto base = ops::RandomUniformInt(scope, tf_shape(rank, dims), 0, maxval);
    return ops::Cast(scope, base, dtype);
  }

  if (DataTypeIsComplex(dtype)) {
    if (dtype == DT_COMPLEX64) {
      auto real = ops::RandomNormal(scope, tf_shape(rank, dims), DT_FLOAT);
      auto imag = ops::RandomNormal(scope, tf_shape(rank, dims), DT_FLOAT);
      ops::Complex::Attrs attrs;
      return ops::Complex(scope, real, imag, attrs.Tout(dtype));
    } else {  // dtype == DT_COMPLEX128
      auto real = ops::RandomNormal(scope, tf_shape(rank, dims), DT_DOUBLE);
      auto imag = ops::RandomNormal(scope, tf_shape(rank, dims), DT_DOUBLE);
      ops::Complex::Attrs attrs;
      return ops::Complex(scope, real, imag, attrs.Tout(dtype));
    }
  }

  if (DataTypeIsQuantized(dtype)) {
    auto base = ops::RandomUniform(scope, tf_shape(rank, dims), DT_FLOAT);
    return ops::QuantizeV2(scope, base, 0.0f, 1.0f, dtype).output;
  }

  if (dtype == DT_STRING) {
    auto base = ops::RandomNormal(scope, tf_shape(rank, dims), DT_FLOAT);
    return ops::AsString(scope, base);
  }

  if (dtype == DT_BOOL) {
    auto left = ops::RandomNormal(scope, tf_shape(rank, dims), DT_FLOAT);
    auto right = ops::RandomNormal(scope, tf_shape(rank, dims), DT_FLOAT);
    return ops::Less(scope, left, right);
  }

  assert(false);
}

tensorflow::Input tf_int_tensor(const ::tensorflow::Scope& scope,
                                DataType dtype, size_t rank, int* dims,
                                size_t int_array_size, int* int_array) {
  assert(DataTypeIsInteger(dtype));
  assert(int_array_size <= ARRAY_SIZE_MAX);

  if (rank == 0) {
    Input base = int_array[0];
    return ops::Cast(scope, base, dtype);
  }

  if (rank == 1) {
    if (int_array_size == 0) {
      return Tensor(dtype, TensorShape({0}));
    } else if (int_array_size == 1) {
      Input base = {int_array[0]};
      return ops::Cast(scope, base, dtype);
    } else if (int_array_size == 2) {
      Input base = {int_array[0], int_array[1]};
      return ops::Cast(scope, base, dtype);
    } else if (int_array_size == 3) {
      Input base = {int_array[0], int_array[1], int_array[2]};
      return ops::Cast(scope, base, dtype);
    } else if (int_array_size == 4) {
      Input base = {int_array[0], int_array[1], int_array[2], int_array[3]};
      return ops::Cast(scope, base, dtype);
    } else if (int_array_size == 5) {
      Input base = {int_array[0], int_array[1], int_array[2], int_array[3],
                    int_array[4]};
      return ops::Cast(scope, base, dtype);
    } else if (int_array_size == 6) {
      Input base = {int_array[0], int_array[1], int_array[2],
                    int_array[3], int_array[4], int_array[5]};
      return ops::Cast(scope, base, dtype);
    }
    assert(false);
  }

  return tf_tensor(scope, false, dtype, rank, dims);
}

std::vector<DataType>& dtype_dict() {
  static std::vector<DataType> dtype_dict_ = {
      // tensorflow/core/framework/types.proto

      // flaoting
      DT_HALF, DT_BFLOAT16, DT_FLOAT, DT_DOUBLE, DT_FLOAT8_E4M3FN,
      DT_FLOAT8_E5M2,

      // integer
      DT_INT4, DT_UINT4, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32,
      DT_UINT32, DT_INT64, DT_UINT64,

      // complex
      DT_COMPLEX64, DT_COMPLEX128,

      // quantized
      DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32,

      DT_STRING, DT_BOOL,

      // DT_RESOURCE,
      // DT_VARIANT,
  };
  return dtype_dict_;
}

DataType get_dtype(size_t i) { return dtype_dict()[i]; }

std::vector<double>& double_dict() {
  static std::vector<double> double_dict_ = {
      -1.0,
      0.0,
      1e-20,
      1e-12,
      1e-8,
      1e-6,
      1e-5,
      1e-4,
      1e-2,
      0.1,
      1.0 / 8.0,
      0.25,
      1.0 / 3.0,
      0.5,
      0.75,
      1.0,
      2.0,
      4.0,
      20.0,
      std::numeric_limits<double>::min(),
      std::numeric_limits<double>::max(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN(),
  };
  return double_dict_;
}

double get_double(size_t i) { return double_dict()[i]; }

std::vector<float>& float_dict() {
  static std::vector<float> float_dict_ = {
      -1.0,
      0.0,
      1e-20,
      1e-12,
      1e-8,
      1e-6,
      1e-5,
      1e-4,
      1e-2,
      0.1,
      1.0 / 8.0,
      0.25,
      1.0 / 3.0,
      0.5,
      0.75,
      1.0,
      2.0,
      4.0,
      20.0,
      std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(),
  };
  return float_dict_;
}

float get_float(size_t i) { return float_dict()[i]; }

const long TENSOR_NUMEL_MAX = 1 << 15;

std::optional<long> safe_mult(long a, long b) {
  if (a == 0)
    return 0;
  else if (b == 0)
    return 0;

  long result = a * b;
  if (a != result / b)
    return std::nullopt;
  else
    return result;
}

bool is_too_big(std::vector<long> shape) {
  if (shape.size() == 0) return false;

  long acc = shape[0];
  for (size_t i = 1; i < shape.size(); i++) {
    std::optional<long> res = safe_mult(acc, shape[i]);
    if (!res.has_value()) return true;
    acc = res.value();
  }
  if (acc >= TENSOR_NUMEL_MAX)
    return true;
  else
    return false;
}

bool is_too_big(size_t rank, std::vector<long> shape) {
  assert(rank <= shape.size());
  long acc = 1;
  for (size_t i = 0; i < (size_t)rank; i++) {
    std::optional<long> res = safe_mult(acc, shape[i]);
    if (!res.has_value()) return true;
    acc = res.value();
  }
  if (acc == LONG_MIN || std::abs(acc) >= TENSOR_NUMEL_MAX)
    return true;
  else
    return false;
}

}  // namespace fuzzer_util
