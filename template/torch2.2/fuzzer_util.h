#ifndef FUZZER_UTIL
#define FUZZER_UTIL

#include <torch/torch.h>

#include <cassert>
#include <limits>
#include <optional>

namespace fuzzer_util {

int abort_if_pytorch_internal_assertion_failed(std::string msg);
int abort_if_not_expected_exception(std::string msg);

std::vector<torch::Dtype>& scalar_dtype_dict();
torch::Dtype get_scalar_dtype(size_t i);

std::vector<torch::Dtype>& dtype_dict();
torch::Dtype get_dtype(size_t i);

torch::Tensor torch_tensor(torch::Dtype dtype, size_t rank,
                           std::vector<long> shape_);
torch::Tensor torch_tensor(torch::Dtype dtype, torch::Layout layout,
                           size_t rank, std::vector<long> shape);

c10::Scalar torch_scalar(
    torch::Dtype dtype, long intValue, unsigned short unsignedIntValue,
    c10::BFloat16 bFloatValue, c10::Half halfValue, float floatValue,
    double doubleValue, c10::Half complex32RealValue,
    c10::Half complex32ImaginaryValue, float complex64RealValue,
    float complex64ImaginaryValue, float complex128RealValue,
    float complex128ImaginaryValue, bool boolValue);

std::vector<const char*>& string_dict();
const char* get_string(size_t i);

std::vector<double>& double_dict();
double get_double(size_t i);

std::vector<float>& float_dict();
float get_float(size_t i);

std::vector<c10::Half>& half_dict();
c10::Half get_half(size_t i);

std::vector<c10::BFloat16>& bfloat_dict();
c10::BFloat16 get_bfloat(size_t i);

std::vector<c10::MemoryFormat>& memory_format_dict();
c10::MemoryFormat get_memory_format(size_t i);

std::vector<c10::Layout>& layout_dict();
c10::Layout get_layout(size_t i);

std::vector<c10::DeviceType>& device_dict();
c10::DeviceType get_device(size_t i);

template <typename T>
std::vector<T> vector_init(size_t size, std::vector<T> vec) {
  assert(size <= vec.size());
  return std::vector<T>(&vec[0], &vec[size]);
}

template <typename T>
c10::ArrayRef<T> arrayref_init(size_t size, std::vector<T> arr) {
  assert(size <= arr.size());
  return c10::ArrayRef<T>(std::vector<T>(&arr[0], &arr[size]));
}

template <size_t D, typename T = int64_t>
torch::ExpandingArrayWithOptionalElem<D, T> expandingarray_with_optional_elem(
    std::vector<c10::optional<T>> arr) {
  assert(arr.size() == D);
  return torch::ExpandingArrayWithOptionalElem<D, T>(arr);
}

bool is_too_big(size_t rank, std::vector<long> shape);

}  // namespace fuzzer_util

#endif
