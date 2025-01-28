#include "torch_symarg.h"

#include <cassert>

#include "options.h"

// Should be consistent with
// <TORCH_HOME>/test/cpp/fuzzing/fuzzer_util.h
const size_t TORCH_MAX_RANK = 5;
const size_t TORCH_MAX_VECTOR_SIZE = 6;
const size_t TORCH_MAX_ARRAYREF_SIZE = 6;

TorchIntSymArg::TorchIntSymArg(std::string name_, std::string specifier_)
    : TorchNumSymArg(name_), specifier(specifier_) {}
void TorchIntSymArg::set_default(int value) { default_value = value; }

std::string TorchIntSymArg::type() const { return specifier; }
std::string TorchIntSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return bracket(type()) + bracket(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return std::to_string(concrete_val.value());
  } else {
    UNREACHABLE;
  }
}

std::vector<std::string> TorchIntSymArg::gen_soft_constraint() const {
  int min = default_value.has_value() ? default_value.value() : 0;
  return {setup_var(name) + gte + std::to_string(min)};
}

TorchSymIntSymArg::TorchSymIntSymArg(std::string name_) : TorchSymArg(name_) {
  intsymarg = std::make_unique<TorchIntSymArg>(name + "_int", "long");
}
void TorchSymIntSymArg::set_default(int value) {
  intsymarg->set_default(value);
}

std::string TorchSymIntSymArg::type() const { return "c10::SymInt"; }
std::string TorchSymIntSymArg::initializer(GEN_MODE mode) const {
  return type() + bracket(intsymarg->expr(mode));
}

std::vector<std::string> TorchSymIntSymArg::gen_arg_setup() const {
  return intsymarg->gen_arg_setup();
}
std::vector<std::string> TorchSymIntSymArg::gen_soft_constraint() const {
  return intsymarg->gen_soft_constraint();
}

TorchUnsignedIntSymArg::TorchUnsignedIntSymArg(std::string name_)
    : TorchNumSymArg(name_) {}
void TorchUnsignedIntSymArg::set_default(unsigned int value) {
  default_value = value;
}

std::string TorchUnsignedIntSymArg::type() const { return "unsigned long"; }
std::string TorchUnsignedIntSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return callback_var(name);
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return std::to_string(concrete_val.value());
  } else {
    UNREACHABLE;
  }
}

std::vector<std::string> TorchUnsignedIntSymArg::gen_hard_constraint() const {
  return {setup_var(name) + gte + std::to_string(0)};
}
std::vector<std::string> TorchUnsignedIntSymArg::gen_soft_constraint() const {
  if (default_value.has_value())
    return {setup_var(name) + gte + std::to_string(default_value.value())};
  else
    return {};
}

TorchBoundedIntSymArg::TorchBoundedIntSymArg(std::string name_, size_t size_)
    : TorchBoundedSymArg(name_, 0, size_) {}
std::string TorchBoundedIntSymArg::type() const { return "size_t"; }
std::string TorchBoundedIntSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return bracket(type()) + bracket(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return std::to_string(concrete_val.value());
  } else {
    UNREACHABLE;
  }
}

TorchBoolSymArg::TorchBoolSymArg(std::string name_)
    : TorchBoundedSymArg(name_, std::vector<std::string>({"false", "true"})) {}

std::string TorchBoolSymArg::type() const { return "bool"; }
std::string TorchBoolSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return bracket(type()) + bracket(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return (bool)(concrete_val.value()) ? "true" : "false";
  } else {
    UNREACHABLE;
  }
}

TorchStringSymArg::TorchStringSymArg(std::string name_, bool is_view_)
    : TorchBoundedSymArg(name_, "string_dict().size()"), is_view(is_view_) {}

std::string TorchStringSymArg::type() const {
  if (is_view) return "c10::string_view";

  return "std::string";
}
std::string TorchStringSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return "get_string" + bracket(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return "get_string" + bracket(std::to_string(concrete_val.value()));
  } else {
    UNREACHABLE;
  }
}

TorchBFloatSymArg::TorchBFloatSymArg(std::string name_)
    : TorchBoundedSymArg(name_, "bfloat_dict().size()") {}

std::string TorchBFloatSymArg::type() const { return "__bf16"; }
std::string TorchBFloatSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return "get_bfloat" + bracket(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return "get_bfloat" + bracket(std::to_string(concrete_val.value()));
  } else {
    UNREACHABLE;
  }
}

TorchHalfSymArg::TorchHalfSymArg(std::string name_)
    : TorchBoundedSymArg(name_, "half_dict().size()") {}

std::string TorchHalfSymArg::type() const { return "__fp16"; }
std::string TorchHalfSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return "get_half" + bracket(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return "get_half" + bracket(std::to_string(concrete_val.value()));
  } else {
    UNREACHABLE;
  }
}

TorchFloatSymArg::TorchFloatSymArg(std::string name_)
    : TorchBoundedSymArg(name_, "float_dict().size()") {}

std::string TorchFloatSymArg::type() const { return "float"; }
std::string TorchFloatSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return "get_float" + bracket(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return "get_float" + bracket(std::to_string(concrete_val.value()));
  } else {
    UNREACHABLE;
  }
}

TorchDoubleSymArg::TorchDoubleSymArg(std::string name_)
    : TorchBoundedSymArg(name_, "double_dict().size()") {}

std::string TorchDoubleSymArg::type() const { return "double"; }
std::string TorchDoubleSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return "get_double" + bracket(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return "get_double" + bracket(std::to_string(concrete_val.value()));
  } else {
    UNREACHABLE;
  }
}

TorchMemoryFormatSymArg::TorchMemoryFormatSymArg(std::string name_)
    : TorchBoundedSymArg(name_, "memory_format_dict().size()") {}

std::string TorchMemoryFormatSymArg::type() const {
  return "c10::MemoryFormat";
}
std::string TorchMemoryFormatSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return "get_memory_format" + bracket(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return "get_memory_format" + bracket(std::to_string(concrete_val.value()));
  } else {
    UNREACHABLE;
  }
}

TorchLayoutSymArg::TorchLayoutSymArg(std::string name_)
    : TorchBoundedSymArg(name_, "layout_dict().size()") {}

std::string TorchLayoutSymArg::type() const { return "c10::Layout"; }
std::string TorchLayoutSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return "get_layout" + bracket(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return "get_layout" + bracket(std::to_string(concrete_val.value()));
  } else {
    UNREACHABLE;
  }
}

TorchDeviceSymArg::TorchDeviceSymArg(std::string name_)
    : TorchBoundedSymArg(name_, "device_dict().size()") {}

std::string TorchDeviceSymArg::type() const { return "c10::Device"; }
std::string TorchDeviceSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return "get_device" + bracket(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return "get_device" + bracket(std::to_string(concrete_val.value()));
  } else {
    UNREACHABLE;
  }
}

TorchDtypeSymArg::TorchDtypeSymArg(std::string name_)
    : TorchBoundedSymArg(name_, "dtype_dict().size()") {}

std::string TorchDtypeSymArg::type() const { return "c10::ScalarType"; }
std::string TorchDtypeSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return "get_dtype" + bracket(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return "get_dtype" + bracket(std::to_string(concrete_val.value()));
  } else {
    UNREACHABLE;
  }
}

TorchScalarDtypeSymArg::TorchScalarDtypeSymArg(std::string name_)
    : TorchBoundedSymArg(name_, "scalar_dtype_dict().size()") {}

std::string TorchScalarDtypeSymArg::type() const { return "c10::ScalarType"; }
std::string TorchScalarDtypeSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return "get_scalar_dtype" + bracket(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return "get_scalar_dtype" + bracket(std::to_string(concrete_val.value()));
  } else {
    UNREACHABLE;
  }
}

TorchVariantSymArg::TorchVariantSymArg(
    std::string name_, std::vector<std::unique_ptr<TorchSymArg>> symargs_)
    : TorchBoundedSymArg(name_, get_names(symargs_)) {
  symargs = std::move(symargs_);
}

std::string TorchVariantSymArg::type() const { return name + "_t"; }
std::string TorchVariantSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return name + square(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return name + square(std::to_string(concrete_val.value()));
  } else {
    UNREACHABLE;
  }
}

void TorchVariantSymArg::resolve_name_conflict(
    std::set<std::string>& names_seen) {
  TorchSymArg::resolve_name_conflict(names_seen);
  for (auto& symarg : symargs) symarg->resolve_name_conflict(names_seen);
}

std::vector<std::string> TorchVariantSymArg::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  concat(arg_setup, TorchBoundedSymArg::gen_arg_setup());
  for (auto& symarg : symargs) concat(arg_setup, symarg->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TorchVariantSymArg::gen_hard_constraint() const {
  std::vector<std::string> hard_ctrs =
      TorchBoundedSymArg::gen_hard_constraint();
  for (auto& symarg : symargs) concat(hard_ctrs, symarg->gen_hard_constraint());
  return hard_ctrs;
}
std::vector<std::string> TorchVariantSymArg::gen_soft_constraint() const {
  std::vector<std::string> soft_ctrs;
  for (auto& symarg : symargs) concat(soft_ctrs, symarg->gen_soft_constraint());
  return soft_ctrs;
}
std::vector<std::string> TorchVariantSymArg::gen_input_pass_condition() const {
  std::vector<std::string> ignore_conds;
  for (auto& symarg : symargs)
    concat(ignore_conds, symarg->gen_input_pass_condition());
  return ignore_conds;
}
std::vector<std::string> TorchVariantSymArg::gen_arg_initialization(
    GEN_MODE mode) const {
  std::vector<std::string> arg_initialization;
  for (auto& symarg : symargs)
    concat(arg_initialization, symarg->gen_arg_initialization(mode));
  concat(arg_initialization, gen_typedef());
  concat(arg_initialization, gen_vector(mode));
  arg_initialization.back() = arg_initialization.back() + newline;
  return arg_initialization;
}

void TorchVariantSymArg::assign_concrete_enum_args(std::vector<long>& args) {
  TorchBoundedSymArg::assign_concrete_enum_args(args);
  for (auto& symarg : symargs) symarg->assign_concrete_enum_args(args);
}
void TorchVariantSymArg::assign_concrete_int_args(std::vector<long>& args) {
  for (auto& symarg : symargs) symarg->assign_concrete_int_args(args);
}

std::vector<std::string> TorchVariantSymArg::gen_typedef() const {
  std::vector<std::string> typedef_str;
  typedef_str.push_back("typedef");
  typedef_str.push_back("  std::variant<");
  for (size_t i = 0; i < symargs.size(); i++) {
    std::string symarg_type = "    " + symargs[i]->type();
    if (i != symargs.size() - 1) symarg_type += comma;
    typedef_str.push_back(symarg_type);
  }
  typedef_str.push_back("  > " + type() + semicolon);
  return typedef_str;
}
std::vector<std::string> TorchVariantSymArg::gen_vector(GEN_MODE mode) const {
  std::vector<std::string> vector_str;
  vector_str.push_back("std::vector<" + type() + "> " + name + " = {");
  for (size_t i = 0; i < symargs.size(); i++) {
    std::string symarg_expr = "  " + symargs[i]->expr(mode);
    if (i != symargs.size() - 1) symarg_expr += comma;
    vector_str.push_back(symarg_expr);
  }
  vector_str.push_back("}" + semicolon);
  return vector_str;
}

TorchEnumSymArg::TorchEnumSymArg(std::string name_, std::string enum_name_)
    : TorchSymArg(name_) {
  enum_name = enum_name_;
}

std::string TorchEnumSymArg::type() const {
  return "torch::enumtype::" + enum_name;
}
std::string TorchEnumSymArg::initializer(GEN_MODE mode) const {
  return "torch::" + enum_name;
}

TorchUnfixedArraySymArg::TorchUnfixedArraySymArg(
    std::string name_, std::vector<std::unique_ptr<TorchSymArg>> symargs_)
    : TorchSymArg(name_) {
  symargs = std::move(symargs_);
  size = std::make_unique<TorchBoundedIntSymArg>(name + "_size",
                                                 symargs.size() + 1);
}
TorchUnfixedArraySymArg::TorchUnfixedArraySymArg(std::string name_,
                                                 std::string base_type_str_)
    : TorchSymArg(name_), base_type_str(base_type_str_) {}

std::string TorchUnfixedArraySymArg::var() const { return name; }

void TorchUnfixedArraySymArg::resolve_name_conflict(
    std::set<std::string>& names_seen) {
  TorchSymArg::resolve_name_conflict(names_seen);
  if (stable()) {
    size->resolve_name_conflict(names_seen);
    for (auto& symarg : symargs) symarg->resolve_name_conflict(names_seen);
  }
}

std::vector<std::string> TorchUnfixedArraySymArg::gen_arg_setup() const {
  if (!stable()) return {};

  std::vector<std::string> arg_setup;
  concat(arg_setup, size->gen_arg_setup());
  for (auto& symarg : symargs) concat(arg_setup, symarg->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TorchUnfixedArraySymArg::gen_hard_constraint() const {
  if (!stable()) return {};

  std::vector<std::string> hard_ctrs;
  concat(hard_ctrs, size->gen_hard_constraint());
  for (auto& symarg : symargs) concat(hard_ctrs, symarg->gen_hard_constraint());
  return hard_ctrs;
}
std::vector<std::string> TorchUnfixedArraySymArg::gen_soft_constraint() const {
  if (!stable()) return {};

  std::vector<std::string> soft_ctrs;
  for (auto& symarg : symargs) concat(soft_ctrs, symarg->gen_soft_constraint());
  return soft_ctrs;
}
std::vector<std::string> TorchUnfixedArraySymArg::gen_arg_initialization(
    GEN_MODE mode) const {
  std::vector<std::string> arg_initialization;
  for (auto& symarg : symargs)
    concat(arg_initialization, symarg->gen_arg_initialization(mode));
  arg_initialization.push_back(type() + space + var() + assign +
                               initializer(mode) + semicolon + newline);
  return arg_initialization;
}

void TorchUnfixedArraySymArg::assign_concrete_enum_args(
    std::vector<long>& args) {
  if (!stable()) return;

  size->assign_concrete_enum_args(args);
  for (auto& symarg : symargs) symarg->assign_concrete_enum_args(args);
}
void TorchUnfixedArraySymArg::assign_concrete_int_args(
    std::vector<long>& args) {
  if (!stable()) return;

  for (auto& symarg : symargs) symarg->assign_concrete_int_args(args);
}

bool TorchUnfixedArraySymArg::stable() const { return !symargs.empty(); }
std::string TorchUnfixedArraySymArg::base_type() const {
  if (!stable()) return base_type_str;

  return symargs[0]->type();
}

TorchVectorSymArg::TorchVectorSymArg(
    std::string name_, std::vector<std::unique_ptr<TorchSymArg>> symargs_)
    : TorchUnfixedArraySymArg(name_, std::move(symargs_)) {}
TorchVectorSymArg::TorchVectorSymArg(std::string name_,
                                     std::string base_type_str_)
    : TorchUnfixedArraySymArg(name_, base_type_str_) {}

std::string TorchVectorSymArg::type() const {
  return "std::vector<" + base_type() + ">";
}
std::string TorchVectorSymArg::initializer(GEN_MODE mode) const {
  if (!stable()) return type() + "({})";

  return "vector_init<" + base_type() + ">" +
         bracket(size->expr(mode) + comma + to_init_list(symargs, mode));
}

TorchArrayRefSymArg::TorchArrayRefSymArg(
    std::string name_, std::vector<std::unique_ptr<TorchSymArg>> symargs_)
    : TorchSymArg(name_) {
  vec = std::make_unique<TorchVectorSymArg>(name + "_vec", std::move(symargs_));
}
TorchArrayRefSymArg::TorchArrayRefSymArg(std::string name_,
                                         std::string base_type_str_)
    : TorchSymArg(name_) {
  vec = std::make_unique<TorchVectorSymArg>(name + "_vec", base_type_str_);
}

std::string TorchArrayRefSymArg::type() const {
  return "c10::ArrayRef<" + vec->base_type() + ">";
}
std::string TorchArrayRefSymArg::var() const { return name; }
std::string TorchArrayRefSymArg::initializer(GEN_MODE mode) const {
  return type() + bracket(vec->expr(mode));
}

void TorchArrayRefSymArg::resolve_name_conflict(
    std::set<std::string>& names_seen) {
  TorchSymArg::resolve_name_conflict(names_seen);
  vec->resolve_name_conflict(names_seen);
}

std::vector<std::string> TorchArrayRefSymArg::gen_arg_setup() const {
  return vec->gen_arg_setup();
}
std::vector<std::string> TorchArrayRefSymArg::gen_hard_constraint() const {
  return vec->gen_hard_constraint();
}
std::vector<std::string> TorchArrayRefSymArg::gen_soft_constraint() const {
  return vec->gen_soft_constraint();
}
std::vector<std::string> TorchArrayRefSymArg::gen_arg_initialization(
    GEN_MODE mode) const {
  std::vector<std::string> arg_initialization;
  concat(arg_initialization, vec->gen_arg_initialization(mode));
  arg_initialization.push_back(type() + space + var() + assign +
                               initializer(mode) + semicolon + newline);
  return arg_initialization;
}

void TorchArrayRefSymArg::assign_concrete_enum_args(std::vector<long>& args) {
  vec->assign_concrete_enum_args(args);
}
void TorchArrayRefSymArg::assign_concrete_int_args(std::vector<long>& args) {
  vec->assign_concrete_int_args(args);
}

TorchFixedArraySymArg::TorchFixedArraySymArg(
    std::string name_, size_t size_,
    std::vector<std::unique_ptr<TorchSymArg>> symargs_)
    : TorchSymArg(name_) {
  size = size_;
  symargs = std::move(symargs_);
  assert(symargs.size() == size);
}

std::string TorchFixedArraySymArg::var() const { return name; }

void TorchFixedArraySymArg::resolve_name_conflict(
    std::set<std::string>& names_seen) {
  TorchSymArg::resolve_name_conflict(names_seen);
  for (auto& symarg : symargs) symarg->resolve_name_conflict(names_seen);
}

std::vector<std::string> TorchFixedArraySymArg::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  for (auto& symarg : symargs) concat(arg_setup, symarg->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TorchFixedArraySymArg::gen_hard_constraint() const {
  std::vector<std::string> hard_ctrs;
  for (auto& symarg : symargs) concat(hard_ctrs, symarg->gen_hard_constraint());
  return hard_ctrs;
}
std::vector<std::string> TorchFixedArraySymArg::gen_soft_constraint() const {
  std::vector<std::string> soft_ctrs;
  for (auto& symarg : symargs) concat(soft_ctrs, symarg->gen_soft_constraint());
  return soft_ctrs;
}
std::vector<std::string> TorchFixedArraySymArg::gen_arg_initialization(
    GEN_MODE mode) const {
  std::vector<std::string> arg_initialization;
  for (auto& symarg : symargs)
    concat(arg_initialization, symarg->gen_arg_initialization(mode));
  arg_initialization.push_back(type() + space + var() + assign +
                               initializer(mode) + semicolon + newline);
  return arg_initialization;
}

void TorchFixedArraySymArg::assign_concrete_enum_args(std::vector<long>& args) {
  for (auto& symarg : symargs) symarg->assign_concrete_enum_args(args);
}
void TorchFixedArraySymArg::assign_concrete_int_args(std::vector<long>& args) {
  for (auto& symarg : symargs) symarg->assign_concrete_int_args(args);
}

TorchExpandingArraySymArg::TorchExpandingArraySymArg(
    std::string name_, size_t size_,
    std::vector<std::unique_ptr<TorchSymArg>> symargs_)
    : TorchFixedArraySymArg(name_, size_, std::move(symargs_)) {}

std::string TorchExpandingArraySymArg::type() const {
  return "torch::ExpandingArray<" + std::to_string(size) + comma +
         symargs[0]->type() + ">";
}
std::string TorchExpandingArraySymArg::initializer(GEN_MODE mode) const {
  return type() + bracket(to_init_list(symargs, mode));
}

TorchExpandingArrayWithOptionalElemSymArg::
    TorchExpandingArrayWithOptionalElemSymArg(
        std::string name_, size_t size_,
        std::vector<std::unique_ptr<TorchSymArg>> symargs_)
    : TorchFixedArraySymArg(name_, size_, std::move(symargs_)) {
  for (auto& symarg : symargs)
    if (dynamic_cast<TorchOptionalSymArg*>(symarg.get()) == nullptr)
      assert(false);
}

std::string TorchExpandingArrayWithOptionalElemSymArg::type() const {
  return "torch::ExpandingArrayWithOptionalElem<" + std::to_string(size) +
         comma + base_type() + ">";
}
std::string TorchExpandingArrayWithOptionalElemSymArg::initializer(
    GEN_MODE mode) const {
  return "expandingarray_with_optional_elem<" + std::to_string(size) + comma +
         base_type() + ">" + bracket(to_init_list(symargs, mode));
}

std::string TorchExpandingArrayWithOptionalElemSymArg::base_type() const {
  if (TorchOptionalSymArg* symarg =
          dynamic_cast<TorchOptionalSymArg*>(symargs[0].get())) {
    return symarg->base_type();
  } else {
    UNREACHABLE;
  }
}

TorchTupleSymArg::TorchTupleSymArg(
    std::string name_, size_t size_,
    std::vector<std::unique_ptr<TorchSymArg>> symargs_)
    : TorchFixedArraySymArg(name_, size_, std::move(symargs_)) {}

std::string TorchTupleSymArg::type() const {
  std::string type_str = "std::tuple<";
  for (size_t i = 0; i < symargs.size(); i++) {
    type_str += symargs[i]->type();
    if (i != symargs.size() - 1) type_str += comma;
  }
  type_str += ">";

  return type_str;
}
std::string TorchTupleSymArg::initializer(GEN_MODE mode) const {
  return type() + bracket(to_init_list(symargs, mode));
}

TorchPairSymArg::TorchPairSymArg(
    std::string name_, std::vector<std::unique_ptr<TorchSymArg>> symargs_)
    : TorchFixedArraySymArg(name_, 2, std::move(symargs_)) {}

std::string TorchPairSymArg::type() const {
  return "std::pair<" + symargs[0]->type() + comma + symargs[1]->type() + ">";
}
std::string TorchPairSymArg::initializer(GEN_MODE mode) const {
  return type() + bracket(to_init_list(symargs, mode));
}

TorchTensorSymArg::TorchTensorSymArg(std::string name_) : TorchSymArg(name_) {
  dtype = std::make_unique<TorchDtypeSymArg>(name + "_dtype");
  rank = std::make_unique<TorchBoundedIntSymArg>(name + "_rank",
                                                 TORCH_MAX_RANK + 1);
  for (size_t i = 0; i < TORCH_MAX_RANK; i++)
    dims.push_back(std::make_unique<TorchIntSymArg>(
        name + "_" + std::to_string(i), "long"));
  for (auto&& dim : dims) dim->set_default(1);
}

std::string TorchTensorSymArg::type() const { return "torch::Tensor"; }
std::string TorchTensorSymArg::var() const { return name; }
std::string TorchTensorSymArg::initializer(GEN_MODE mode) const {
  std::string layout_expr = "";
  std::string args = dtype->expr(mode) + comma + layout_expr +
                     rank->expr(mode) + comma + to_init_list(dims, mode);
  return "torch_tensor" + bracket(args);
}

void TorchTensorSymArg::resolve_name_conflict(
    std::set<std::string>& names_seen) {
  TorchSymArg::resolve_name_conflict(names_seen);
  dtype->resolve_name_conflict(names_seen);
  rank->resolve_name_conflict(names_seen);
  for (auto& dim : dims) dim->resolve_name_conflict(names_seen);
}

std::vector<std::string> TorchTensorSymArg::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  concat(arg_setup, dtype->gen_arg_setup());
  concat(arg_setup, rank->gen_arg_setup());
  for (auto& dim : dims) concat(arg_setup, dim->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TorchTensorSymArg::gen_hard_constraint() const {
  std::vector<std::string> hard_ctrs;
  concat(hard_ctrs, dtype->gen_hard_constraint());
  concat(hard_ctrs, rank->gen_hard_constraint());
  for (auto& dim : dims) concat(hard_ctrs, dim->gen_soft_constraint());
  return hard_ctrs;
}
std::vector<std::string> TorchTensorSymArg::gen_input_pass_condition() const {
  return {"is_too_big" + bracket(rank->expr(MODE_DRIVER) + comma +
                                 to_init_list(dims, MODE_DRIVER))};
}
std::vector<std::string> TorchTensorSymArg::gen_arg_initialization(
    GEN_MODE mode) const {
  return {type() + space + var() + assign + initializer(mode) + semicolon +
          newline};
}

void TorchTensorSymArg::assign_concrete_enum_args(std::vector<long>& args) {
  dtype->assign_concrete_enum_args(args);
  rank->assign_concrete_enum_args(args);
  for (auto& dim : dims) dim->assign_concrete_enum_args(args);
}
void TorchTensorSymArg::assign_concrete_int_args(std::vector<long>& args) {
  for (auto& dim : dims) dim->assign_concrete_int_args(args);
}

TorchScalarSymArg::TorchScalarSymArg(std::string name_) : TorchSymArg(name_) {
  dtype = std::make_unique<TorchScalarDtypeSymArg>(name + "_dtype");
  intValue = std::make_unique<TorchIntSymArg>(name + "_int", "int");
  uintValue = std::make_unique<TorchUnsignedIntSymArg>(name + "_uint");
  bfloatValue = std::make_unique<TorchBFloatSymArg>(name + "_bfloat");
  halfValue = std::make_unique<TorchHalfSymArg>(name + "_half");
  floatValue = std::make_unique<TorchFloatSymArg>(name + "_float");
  doubleValue = std::make_unique<TorchDoubleSymArg>(name + "_double");
  if (DLL_VERSION != "1.11") {
    realValue32 = std::make_unique<TorchHalfSymArg>(name + "_real32");
    imaginaryValue32 = std::make_unique<TorchHalfSymArg>(name + "_imag32");
  }
  realValue64 = std::make_unique<TorchFloatSymArg>(name + "_real64");
  imaginaryValue64 = std::make_unique<TorchFloatSymArg>(name + "_imag64");
  realValue128 = std::make_unique<TorchDoubleSymArg>(name + "_real128");
  imaginaryValue128 = std::make_unique<TorchDoubleSymArg>(name + "_imag128");
  boolValue = std::make_unique<TorchBoolSymArg>(name + "_bool");
  symargs.push_back(dtype.get());
  symargs.push_back(intValue.get());
  symargs.push_back(uintValue.get());
  symargs.push_back(bfloatValue.get());
  symargs.push_back(halfValue.get());
  symargs.push_back(floatValue.get());
  symargs.push_back(doubleValue.get());
  if (DLL_VERSION != "1.11") {
    symargs.push_back(realValue32.get());
    symargs.push_back(imaginaryValue32.get());
  }
  symargs.push_back(realValue64.get());
  symargs.push_back(imaginaryValue64.get());
  symargs.push_back(realValue128.get());
  symargs.push_back(imaginaryValue128.get());
  symargs.push_back(boolValue.get());
}

std::string TorchScalarSymArg::type() const { return "c10::Scalar"; }
std::string TorchScalarSymArg::var() const { return name; }
std::string TorchScalarSymArg::initializer(GEN_MODE mode) const {
  std::string candidates;
  for (size_t i = 0; i < symargs.size(); i++) {
    candidates += symargs[i]->expr(mode);
    if (i != symargs.size() - 1) candidates += comma;
  }
  return "torch_scalar" + bracket(candidates);
}

void TorchScalarSymArg::resolve_name_conflict(
    std::set<std::string>& names_seen) {
  TorchSymArg::resolve_name_conflict(names_seen);
  for (auto& symarg : symargs) symarg->resolve_name_conflict(names_seen);
}

std::vector<std::string> TorchScalarSymArg::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  for (auto& symarg : symargs) concat(arg_setup, symarg->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TorchScalarSymArg::gen_hard_constraint() const {
  std::vector<std::string> hard_ctrs;
  for (auto& symarg : symargs) concat(hard_ctrs, symarg->gen_hard_constraint());
  return hard_ctrs;
}
std::vector<std::string> TorchScalarSymArg::gen_soft_constraint() const {
  std::vector<std::string> soft_ctrs;
  for (auto& symarg : symargs) concat(soft_ctrs, symarg->gen_soft_constraint());
  return soft_ctrs;
}
std::vector<std::string> TorchScalarSymArg::gen_arg_initialization(
    GEN_MODE mode) const {
  return {type() + space + var() + assign + initializer(mode) + semicolon +
          newline};
}

void TorchScalarSymArg::assign_concrete_enum_args(std::vector<long>& args) {
  for (auto& symarg : symargs) symarg->assign_concrete_enum_args(args);
}
void TorchScalarSymArg::assign_concrete_int_args(std::vector<long>& args) {
  for (auto& symarg : symargs) symarg->assign_concrete_int_args(args);
}

TorchOptionalSymArg::TorchOptionalSymArg(std::string name_,
                                         std::unique_ptr<TorchSymArg> symarg_)
    : TorchSymArg(name_) {
  has_value = std::make_unique<TorchBoolSymArg>(name + "_hasValue");
  symarg = std::move(symarg_);
}
TorchOptionalSymArg::TorchOptionalSymArg(std::string name_,
                                         std::string base_type_str_)
    : TorchSymArg(name_), base_type_str(base_type_str_) {}

std::string TorchOptionalSymArg::type() const {
  return "c10::optional<" + base_type() + ">";
}
std::string TorchOptionalSymArg::var() const { return name; }
std::string TorchOptionalSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    if (!stable()) return type() + bracket("c10::nullopt");

    return has_value->expr(mode) + " ? " + type() +
           bracket(symarg->expr(mode)) + " : " + "c10::nullopt";
  } else if (mode == MODE_POV) {
    return has_value->expr(mode) == "true"
               ? bracket(base_type()) + bracket(symarg->expr(mode))
               : "c10::nullopt";
  } else {
    UNREACHABLE;
  }
}

void TorchOptionalSymArg::resolve_name_conflict(
    std::set<std::string>& names_seen) {
  TorchSymArg::resolve_name_conflict(names_seen);
  if (stable()) {
    has_value->resolve_name_conflict(names_seen);
    symarg->resolve_name_conflict(names_seen);
  }
}

std::vector<std::string> TorchOptionalSymArg::gen_arg_setup() const {
  if (!stable()) return {};

  std::vector<std::string> arg_setup;
  concat(arg_setup, has_value->gen_arg_setup());
  concat(arg_setup, symarg->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TorchOptionalSymArg::gen_hard_constraint() const {
  if (!stable()) return {};

  std::vector<std::string> hard_ctrs;
  concat(hard_ctrs, has_value->gen_hard_constraint());
  concat(hard_ctrs, symarg->gen_hard_constraint());
  return hard_ctrs;
}
std::vector<std::string> TorchOptionalSymArg::gen_soft_constraint() const {
  if (!stable()) return {};

  return symarg->gen_soft_constraint();
}
std::vector<std::string> TorchOptionalSymArg::gen_input_pass_condition() const {
  if (!stable()) return {};

  return symarg->gen_input_pass_condition();
}
std::vector<std::string> TorchOptionalSymArg::gen_arg_initialization(
    GEN_MODE mode) const {
  std::vector<std::string> arg_initialization;
  if (stable())
    concat(arg_initialization, symarg->gen_arg_initialization(mode));
  arg_initialization.push_back(type() + space + var() + assign +
                               initializer(mode) + semicolon + newline);
  return arg_initialization;
}

void TorchOptionalSymArg::assign_concrete_enum_args(std::vector<long>& args) {
  if (!stable()) return;

  has_value->assign_concrete_enum_args(args);
  symarg->assign_concrete_enum_args(args);
}
void TorchOptionalSymArg::assign_concrete_int_args(std::vector<long>& args) {
  if (!stable()) return;

  symarg->assign_concrete_int_args(args);
}

bool TorchOptionalSymArg::stable() const { return symarg != nullptr; }

std::string TorchOptionalSymArg::base_type() const {
  if (!stable()) return base_type_str;

  return symarg->type();
}

TorchAPIOptionsSymArg::TorchAPIOptionsSymArg(
    std::string name_, std::string api_optons_class_name_,
    std::vector<std::unique_ptr<TorchSymArg>> ctor_symargs_,
    std::vector<std::unique_ptr<TorchSymArg>> member_symargs_)
    : TorchSymArg(name_) {
  api_optons_class_name = api_optons_class_name_;
  ctor_symargs = std::move(ctor_symargs_);
  member_symargs = std::move(member_symargs_);
  for (auto& member_symarg : member_symargs)
    member_symarg_setters.push_back(member_symarg->get_name());
}

std::string TorchAPIOptionsSymArg::type() const {
  return api_optons_class_name;
}
std::string TorchAPIOptionsSymArg::var() const { return name; }
std::string TorchAPIOptionsSymArg::initializer(GEN_MODE mode) const {
  PDG_ASSERT(false);
}

void TorchAPIOptionsSymArg::resolve_name_conflict(
    std::set<std::string>& names_seen) {
  TorchSymArg::resolve_name_conflict(names_seen);
  for (auto& symarg : ctor_symargs) symarg->resolve_name_conflict(names_seen);
  for (auto& symarg : member_symargs) symarg->resolve_name_conflict(names_seen);
}

std::vector<std::string> TorchAPIOptionsSymArg::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  for (auto& symarg : ctor_symargs) concat(arg_setup, symarg->gen_arg_setup());
  for (auto& symarg : member_symargs)
    concat(arg_setup, symarg->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TorchAPIOptionsSymArg::gen_hard_constraint() const {
  std::vector<std::string> hard_ctrs;
  for (auto& symarg : ctor_symargs)
    concat(hard_ctrs, symarg->gen_hard_constraint());
  for (auto& symarg : member_symargs)
    concat(hard_ctrs, symarg->gen_hard_constraint());
  return hard_ctrs;
}
std::vector<std::string> TorchAPIOptionsSymArg::gen_soft_constraint() const {
  std::vector<std::string> soft_ctrs;
  for (auto& symarg : ctor_symargs)
    concat(soft_ctrs, symarg->gen_soft_constraint());
  for (auto& symarg : member_symargs)
    concat(soft_ctrs, symarg->gen_soft_constraint());
  return soft_ctrs;
}
std::vector<std::string> TorchAPIOptionsSymArg::gen_input_pass_condition()
    const {
  std::vector<std::string> ignore_conds;
  for (auto& symarg : ctor_symargs)
    concat(ignore_conds, symarg->gen_input_pass_condition());
  for (auto& symarg : member_symargs)
    concat(ignore_conds, symarg->gen_input_pass_condition());
  return ignore_conds;
}
std::vector<std::string> TorchAPIOptionsSymArg::gen_arg_initialization(
    GEN_MODE mode) const {
  std::vector<std::string> arg_initialization;
  for (auto& symarg : ctor_symargs)
    concat(arg_initialization, symarg->gen_arg_initialization(mode));
  for (auto& symarg : member_symargs)
    concat(arg_initialization, symarg->gen_arg_initialization(mode));
  concat(arg_initialization, gen_api_options_init(mode));
  arg_initialization.back() = arg_initialization.back() + newline;
  return arg_initialization;
}

void TorchAPIOptionsSymArg::assign_concrete_enum_args(std::vector<long>& args) {
  for (auto& symarg : ctor_symargs) symarg->assign_concrete_enum_args(args);
  for (auto& symarg : member_symargs) symarg->assign_concrete_enum_args(args);
}
void TorchAPIOptionsSymArg::assign_concrete_int_args(std::vector<long>& args) {
  for (auto& symarg : ctor_symargs) symarg->assign_concrete_int_args(args);
  for (auto& symarg : member_symargs) symarg->assign_concrete_int_args(args);
}

std::vector<std::string> TorchAPIOptionsSymArg::gen_member_symarg_set(
    GEN_MODE mode) const {
  std::vector<std::string> member_symarg_set;
  assert(member_symargs.size() == member_symarg_setters.size());
  for (size_t i = 0; i < member_symargs.size(); i++)
    member_symarg_set.push_back("." + member_symarg_setters[i] +
                                bracket(member_symargs[i]->expr(mode)));
  return member_symarg_set;
}

std::vector<std::string> TorchAPIOptionsSymArg::gen_api_options_init(
    GEN_MODE mode) const {
  std::vector<std::string> api_options_init;

  api_options_init.push_back("auto " + name + assign);
  std::string initializer = "  " + api_optons_class_name + "(";
  for (size_t i = 0; i < ctor_symargs.size(); i++) {
    initializer += ctor_symargs[i]->expr(mode);
    if (i != ctor_symargs.size() - 1) initializer += comma;
  }
  initializer += ")";
  api_options_init.push_back(initializer);
  for (auto& symarg_set : gen_member_symarg_set(mode))
    api_options_init.push_back("    " + symarg_set);
  api_options_init.back() = api_options_init.back() + semicolon;
  return api_options_init;
}
