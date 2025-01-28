#include "fuzzer_util.h"

namespace fuzzer_util {

int abort_if_pytorch_internal_assertion_failed(std::string msg) {
  static std::string pytorch_assert_fail_msg("INTERNAL ASSERT FAILED");
  if (msg.find(pytorch_assert_fail_msg) != std::string::npos) {
    std::cerr << msg << std::endl;
    return -3;  // PATHFINDER_UNEXPECTED_EXCEPTION
  }
  return -2;  // PATHFINDER_EXPECTED_EXCEPTION
}

int abort_if_not_expected_exception(std::string msg) {
  static std::vector<std::string> expected_exception_msgs = {
      "value cannot be converted to type",  // thrown by
                                            // c10/util/TypeCast.cpp::report_overflow()
  };

  for (auto& expected_exception_msg : expected_exception_msgs)
    if (msg.find(expected_exception_msg) != std::string::npos)
      return -2;  // PATHFINDER_EXPECTED_EXCEPTION
  std::cerr << msg << std::endl;
  return -3;  // PATHFINDER_UNEXPECTED_EXCEPTION
}

std::vector<torch::Dtype>& scalar_dtype_dict() {
  static std::vector<torch::Dtype> scalar_dtype_dict_ = {
      torch::kUInt8,          // ScalarType::Byte
      torch::kInt8,           // ScalarType::Char
      torch::kInt16,          // ScalarType::Short
      torch::kInt32,          // ScalarType::Int
      torch::kInt64,          // ScalarType::Long
      torch::kFloat16,        // ScalarType::Half
      torch::kFloat32,        // ScalarType::Float
      torch::kFloat64,        // ScalarType::Double
      torch::kComplexHalf,    // ScalarType::ComplexHalf
      torch::kComplexFloat,   // ScalarType::ComplexFloat
      torch::kComplexDouble,  // ScalarType::ComplexDouble
      torch::kBool,           // ScalarType::Bool
      torch::kBFloat16,       // ScalarType::BFloat16
  };
  return scalar_dtype_dict_;
}

torch::Dtype get_scalar_dtype(size_t i) { return scalar_dtype_dict()[i]; }

std::vector<torch::Dtype>& qint_dtype_dict() {
  static std::vector<torch::Dtype> qint_dtype_dict_ = {
      c10::kQInt8, c10::kQUInt8, c10::kQInt32, c10::kQUInt4x2, c10::kQUInt2x4,
  };
  return qint_dtype_dict_;
}

torch::Dtype get_qint_dtype(size_t i) { return qint_dtype_dict()[i]; }

std::vector<torch::Dtype>& dtype_dict() {
  static bool initialized = false;
  static std::vector<torch::Dtype> dtype_dict_;
  if (!initialized) {
    dtype_dict_.insert(dtype_dict_.end(), scalar_dtype_dict().begin(),
                       scalar_dtype_dict().end());
    dtype_dict_.insert(dtype_dict_.end(), qint_dtype_dict().begin(),
                       qint_dtype_dict().end());
    initialized = true;
  }
  return dtype_dict_;
}

torch::Dtype get_dtype(size_t i) { return dtype_dict()[i]; }

torch::Tensor torch_tensor(torch::Dtype dtype, std::vector<long> shape) {
  torch::TensorOptions toptions = torch::TensorOptions();

  if (isIntegralType(dtype, true)) {
    toptions = toptions.dtype(dtype);
    size_t high;
    switch (dtype) {
      case torch::kUInt8:
        high = (1 << 8) - 1;
        break;
      case torch::kInt8:
        high = (1 << 7) - 1;
        break;
      case torch::kInt16:
      case torch::kInt32:
      case torch::kInt64:
        high = (1 << 15) - 1;
        break;
      case torch::kBool:
        high = 1;
        break;
      default:
        assert(false);
    }
    return torch::randint(high, shape, toptions);
  } else if (isFloatingType(dtype) || isComplexType(dtype)) {
    toptions = toptions.dtype(dtype);
    return torch::randn(shape, toptions);
  } else if (isQIntType(dtype)) {
    toptions = toptions.dtype(torch::kFloat32);
    auto base = torch::randn(shape, toptions);
    return at::quantize_per_tensor(base, 0.1, 10, dtype);
  } else {
    assert(false);
    return torch::randn({}, toptions);
  }
}

torch::Tensor set_layout(torch::Layout layout, torch::Tensor tensor) {
  switch (layout) {
    case c10::Layout::Strided:
      return tensor;
    case c10::Layout::Sparse:
    case c10::Layout::SparseCsr:
    case c10::Layout::SparseCsc:
    case c10::Layout::SparseBsr:
    case c10::Layout::SparseBsc:
      return tensor.to_sparse(layout);  // COO only
    case c10::Layout::Mkldnn:
      return at::hasMKLDNN() ? tensor.to_mkldnn() : tensor;
    default: {
      assert(false);
      return tensor;
    }
  }
}

torch::Tensor torch_tensor(torch::Dtype dtype, size_t rank,
                           std::vector<long> shape_) {
  assert(rank <= shape_.size());
  std::vector<long> shape(&shape_[0], &shape_[rank]);
  return torch_tensor(dtype, shape);
}

torch::Tensor torch_tensor(torch::Dtype dtype, torch::Layout layout,
                           size_t rank, std::vector<long> shape) {
  auto t = torch_tensor(dtype, rank, shape);
  return set_layout(layout, t);
}

c10::Scalar torch_scalar(
    torch::Dtype dtype, long intValue, unsigned short unsignedIntValue,
    c10::BFloat16 bFloatValue, c10::Half halfValue, float floatValue,
    double doubleValue, c10::Half complex32RealValue,
    c10::Half complex32ImaginaryValue, float complex64RealValue,
    float complex64ImaginaryValue, float complex128RealValue,
    float complex128ImaginaryValue, bool boolValue) {
  switch (dtype) {
    case torch::kInt8:
    case torch::kInt16:
    case torch::kInt32:
    case torch::kInt64:
      return c10::Scalar(intValue);  // int
    case torch::kUInt8:
      return c10::Scalar(unsignedIntValue);  // unsigned int
    case torch::kBFloat16:
      return c10::Scalar(bFloatValue);  // bfloat
    case torch::kFloat16:
      return c10::Scalar(halfValue);  // half
    case torch::kFloat32:
      return c10::Scalar(floatValue);  // float
    case torch::kFloat64:
      return c10::Scalar(doubleValue);  // double
    case torch::kComplexHalf:
      return c10::complex<c10::Half>(complex32RealValue,
                                     complex32ImaginaryValue);  // complexHalf
    case torch::kComplexFloat:
      return c10::complex<float>(complex64RealValue,
                                 complex64ImaginaryValue);  // complexFloat
    case torch::kComplexDouble:
      return c10::complex<double>(complex128RealValue,
                                  complex128ImaginaryValue);  // complexDouble
    case torch::kBool:
      return c10::Scalar(boolValue);  // bool
    default: {
      assert(false);
      return c10::Scalar(false);
    }
  }
}

std::vector<const char*>& string_dict() {
  static std::vector<const char*> string_dict_ = {
      "",        "trunc",    "floor",    "ij",      "xy",        "linear",
      "lower",   "higher",   "midpoint", "nearest", "add",       "multiply",
      "left",    "right",    "constant", "reflect", "replicate", "circular",
      "L",       "U",        "gelsy",    "gels",    "fro",       "nuc",
      "reduced", "complete", "r",        "None",    "gesvd",     "gesvdj",
      "gesvda",  "gelsy",    "getsd",    "gelss",   "forward",   "backward",
      "ortho",   "none",     "tanh",     "same",    "valid",     "\u2208",
  };
  return string_dict_;
}

const char* get_string(size_t i) { return string_dict()[i]; }

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
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
  };
  return float_dict_;
}

float get_float(size_t i) { return float_dict()[i]; }

std::vector<c10::Half>& half_dict() {
  static std::vector<c10::Half> half_dict_ = {
      -1.0,
      0.0,
      1e+4,
      1e+5,
      1e+6,
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
      -65504,
      65504,
      c10::Half(std::numeric_limits<float>::infinity()),
      c10::Half(std::numeric_limits<float>::quiet_NaN()),
  };
  return half_dict_;
}

c10::Half get_half(size_t i) { return half_dict()[i]; }

std::vector<c10::BFloat16>& bfloat_dict() {
  static std::vector<c10::BFloat16> bfloat_dict_ = {
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
      c10::BFloat16(std::numeric_limits<float>::min()),
      c10::BFloat16(std::numeric_limits<float>::max()),
      c10::BFloat16(std::numeric_limits<float>::infinity()),
      c10::BFloat16(std::numeric_limits<float>::quiet_NaN()),
  };
  return bfloat_dict_;
}

c10::BFloat16 get_bfloat(size_t i) { return bfloat_dict()[i]; }

std::vector<c10::MemoryFormat>& memory_format_dict() {
  static std::vector<c10::MemoryFormat> memory_format_dict_ = {
      c10::MemoryFormat::Contiguous,
      c10::MemoryFormat::Preserve,
      c10::MemoryFormat::ChannelsLast,
      c10::MemoryFormat::ChannelsLast3d,
  };
  return memory_format_dict_;
}

c10::MemoryFormat get_memory_format(size_t i) {
  return memory_format_dict()[i];
}

std::vector<c10::Layout>& layout_dict() {
  static std::vector<c10::Layout> layout_dict_ = {
      c10::Layout::Strided,   c10::Layout::Sparse,    c10::Layout::SparseCsr,
      c10::Layout::Mkldnn,    c10::Layout::SparseCsc, c10::Layout::SparseBsr,
      c10::Layout::SparseBsc,
  };
  return layout_dict_;
}

c10::Layout get_layout(size_t i) { return layout_dict()[i]; }

std::vector<c10::DeviceType>& device_dict() {
  static std::vector<c10::DeviceType> device_dict_ = {
      c10::DeviceType::CPU,         c10::DeviceType::CUDA,
      c10::DeviceType::HIP,         c10::DeviceType::FPGA,
      c10::DeviceType::ORT,         c10::DeviceType::XLA,
      c10::DeviceType::Vulkan,      c10::DeviceType::Metal,
      c10::DeviceType::XPU,         c10::DeviceType::MPS,
      c10::DeviceType::Meta,        c10::DeviceType::HPU,
      c10::DeviceType::VE,          c10::DeviceType::Lazy,
      c10::DeviceType::IPU,         c10::DeviceType::MTIA,
      c10::DeviceType::PrivateUse1,

      // c10/core/TensorOptions.h:653
      // c10::DeviceType::MKLDNN,
      // c10::DeviceType::OPENGL,
      // c10::DeviceType::OPENCL,
      // c10::DeviceType::IDEEP,
  };
  return device_dict_;
}

c10::DeviceType get_device(size_t i) { return device_dict()[i]; }

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
