#include "tf_symarg.h"

#include <cassert>

#include "options.h"
#include "tf_datatype.h"

// Should be consistent with
// <TENSORFLOW_HOME>/tensorflow/core/kernels/pathfinder/fuzzer_util.h
const size_t TF_MAX_RANK = 5;
const size_t TF_MAX_ARRAY_SIZE = 6;

TFScopeSymArg::TFScopeSymArg() : TFSymArg(tf_scope_var) {}

std::string TFScopeSymArg::type() const { return "Scope"; }
std::string TFScopeSymArg::var() const { return name; }
std::string TFScopeSymArg::initializer(GEN_MODE mode) const {
  return "Scope::NewRootScope()";
}

std::vector<std::string> TFScopeSymArg::gen_arg_initialization(
    GEN_MODE mode) const {
  return {type() + space + var() + assign + initializer(mode) + semicolon +
          newline};
}
void TFScopeSymArg::resolve_name_conflict(std::set<std::string>& names_seen) {
  return;
}

TFIntSymArg::TFIntSymArg(std::string name_, int min_)
    : TFNumSymArg(name_), min(min_) {}

std::string TFIntSymArg::type() const { return "int"; }
std::string TFIntSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return bracket(type()) + bracket(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return std::to_string(concrete_val.value());
  } else {
    UNREACHABLE;
  }
}

std::vector<std::string> TFIntSymArg::gen_soft_constraint() const {
  return {setup_var(name) + gte + std::to_string(min)};
}

TFBoundedIntSymArg::TFBoundedIntSymArg(std::string name_, int start_,
                                       size_t size_)
    : TFBoundedSymArg(name_, start_, size_) {}

std::string TFBoundedIntSymArg::type() const { return "size_t"; }
std::string TFBoundedIntSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return bracket(type()) + bracket(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return std::to_string(concrete_val.value());
  } else {
    UNREACHABLE;
  }
}

int TFBoundedIntSymArg::min() const {
  assert(start_size_pair.has_value());
  return start_size_pair.value().first;
}
int TFBoundedIntSymArg::max() const {
  assert(start_size_pair.has_value());
  size_t size = start_size_pair.value().second;
  return min() + (int)size - 1;
}

TFBoolSymArg::TFBoolSymArg(std::string name_)
    : TFBoundedSymArg(name_, std::vector<std::string>({"false", "true"})) {}

std::string TFBoolSymArg::type() const { return "bool"; }
std::string TFBoolSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return bracket(type()) + bracket(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return (bool)(concrete_val.value()) ? "true" : "false";
  } else {
    UNREACHABLE;
  }
}

TFFloatSymArg::TFFloatSymArg(std::string name_)
    : TFBoundedSymArg(name_, "float_dict().size()") {}

std::string TFFloatSymArg::type() const { return "float"; }
std::string TFFloatSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return "get_float" + bracket(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return "get_float" + bracket(std::to_string(concrete_val.value()));
  } else {
    UNREACHABLE;
  }
}

TFDoubleSymArg::TFDoubleSymArg(std::string name_)
    : TFBoundedSymArg(name_, "double_dict().size()") {}

std::string TFDoubleSymArg::type() const { return "double"; }
std::string TFDoubleSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return "get_double" + bracket(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return "get_double" + bracket(std::to_string(concrete_val.value()));
  } else {
    UNREACHABLE;
  }
}

TFDtypeSymArg::TFDtypeSymArg(std::string name_,
                             const std::vector<std::string>& allowed_)
    : TFBoundedSymArg(name_, allowed_) {}

std::string TFDtypeSymArg::type() const { return "DataType"; }
std::string TFDtypeSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return allowed_list_name() + square(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return allowed_list_name() + square(std::to_string(concrete_val.value()));
  } else {
    UNREACHABLE;
  }
}
std::vector<std::string> TFDtypeSymArg::gen_arg_initialization(
    GEN_MODE mode) const {
  return {type() + space + allowed_list_name() + "[]" + assign +
          curly(join_strs(value_names)) + semicolon + newline};
}

bool TFDtypeSymArg::is_int_datatype() const {
  return is_integer_datatype(value_names);
}

std::string TFDtypeSymArg::allowed_list_name() const {
  return name + "_allowed";
}

TFStringSymArg::TFStringSymArg(std::string name_,
                               const std::vector<std::string>& allowed_)
    : TFBoundedSymArg(name_, allowed_) {}

std::string TFStringSymArg::type() const {
  if (tstring) return "::tensorflow::tstring";
  return "StringPiece";
}
std::string TFStringSymArg::initializer(GEN_MODE mode) const {
  if (mode == MODE_DRIVER) {
    return allowed_list_name() + square(callback_var(name));
  } else if (mode == MODE_POV) {
    PDG_ASSERT(concrete_val.has_value());
    return allowed_list_name() + square(std::to_string(concrete_val.value()));
  } else {
    UNREACHABLE;
  }
}
std::vector<std::string> TFStringSymArg::gen_arg_initialization(
    GEN_MODE mode) const {
  std::string joined;
  for (size_t i = 0; i < value_names.size(); i++) {
    joined += quoted(value_names[i]);
    if (i != value_names.size() - 1) joined += comma;
  }
  return {"const char*" + space + allowed_list_name() + "[]" + assign +
          curly(joined) + semicolon + newline};
}

std::string TFStringSymArg::allowed_list_name() const {
  return name + "_allowed";
}
void TFStringSymArg::to_tstring() { tstring = true; }

TFArraySymArg::TFArraySymArg(std::string name_,
                             std::vector<std::unique_ptr<TFSymArg>> symargs_)
    : TFSymArg(name_) {
  symargs = std::move(symargs_);
}

std::string TFArraySymArg::type() const { return base_type() + "*"; }
std::string TFArraySymArg::var() const { return name; }
std::string TFArraySymArg::initializer(GEN_MODE mode) const {
  return to_init_list(symargs, mode);
}

void TFArraySymArg::resolve_name_conflict(std::set<std::string>& names_seen) {
  TFSymArg::resolve_name_conflict(names_seen);
  for (auto& symarg : symargs) symarg->resolve_name_conflict(names_seen);
}

std::vector<std::string> TFArraySymArg::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  for (auto& symarg : symargs) concat(arg_setup, symarg->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TFArraySymArg::gen_hard_constraint() const {
  std::vector<std::string> hard_ctrs;
  for (auto& symarg : symargs) concat(hard_ctrs, symarg->gen_hard_constraint());
  return hard_ctrs;
}
std::vector<std::string> TFArraySymArg::gen_soft_constraint() const {
  std::vector<std::string> soft_ctrs;
  for (auto& symarg : symargs) concat(soft_ctrs, symarg->gen_soft_constraint());
  return soft_ctrs;
}
std::vector<std::string> TFArraySymArg::gen_input_pass_condition() const {
  std::vector<std::string> ignore_conds;
  for (auto& symarg : symargs)
    concat(ignore_conds, symarg->gen_input_pass_condition());
  return ignore_conds;
}
std::vector<std::string> TFArraySymArg::gen_arg_initialization(
    GEN_MODE mode) const {
  std::vector<std::string> arg_initialization;
  for (auto& symarg : symargs)
    concat(arg_initialization, symarg->gen_arg_initialization(mode));
  arg_initialization.push_back(base_type() + space + var() + "[]" + assign +
                               initializer(mode) + semicolon + newline);
  tie_strs(arg_initialization);
  return arg_initialization;
}

void TFArraySymArg::assign_concrete_enum_args(std::vector<long>& args) {
  for (auto& symarg : symargs) symarg->assign_concrete_enum_args(args);
}
void TFArraySymArg::assign_concrete_int_args(std::vector<long>& args) {
  for (auto& symarg : symargs) symarg->assign_concrete_int_args(args);
}

std::string TFArraySymArg::base_type() const {
  assert(symargs[0] != nullptr);
  return symargs[0]->type();
}

TFArraySliceSymArg::TFArraySliceSymArg(
    std::string name_, std::vector<std::unique_ptr<TFSymArg>> symargs_,
    size_t size_min, size_t size_max)
    : TFSymArg(name_) {
  assert(size_min <= size_max);
  if (size_min == size_max) {
    assert(symargs_.size() == size_min);
    const_size = symargs_.size();
  } else {
    size_owned = std::make_unique<TFBoundedIntSymArg>(name + "_size", size_min,
                                                      size_max - size_min + 1);
  }
  array = std::make_unique<TFArraySymArg>(name + "_array", std::move(symargs_));
}
TFArraySliceSymArg::TFArraySliceSymArg(
    std::string name_, std::vector<std::unique_ptr<TFSymArg>> symargs_,
    TFBoundedIntSymArg* size_dependent_)
    : TFSymArg(name_) {
  assert(symargs_.size() == size_dependent_->max());
  size_dependent = size_dependent_;
  array = std::make_unique<TFArraySymArg>(name + "_array", std::move(symargs_));
}

std::string TFArraySliceSymArg::type() const {
  assert(array != nullptr);
  return "gtl::ArraySlice<" + array->base_type() + ">";
}
std::string TFArraySliceSymArg::var() const { return name; }
std::string TFArraySliceSymArg::initializer(GEN_MODE mode) const {
  if (const_size.has_value()) {
    return type() + bracket(array->expr(mode));
  } else {
    std::string size_expr = size_owned != nullptr ? size_owned->expr(mode)
                                                  : size_dependent->expr(mode);
    return type() + bracket(array->expr(mode) + comma + size_expr);
  }
}

void TFArraySliceSymArg::resolve_name_conflict(
    std::set<std::string>& names_seen) {
  assert(array != nullptr);

  TFSymArg::resolve_name_conflict(names_seen);
  if (size_owned != nullptr) size_owned->resolve_name_conflict(names_seen);
  array->resolve_name_conflict(names_seen);
}

std::vector<std::string> TFArraySliceSymArg::gen_arg_setup() const {
  assert(array != nullptr);
  std::vector<std::string> arg_setup;
  if (size_owned != nullptr) concat(arg_setup, size_owned->gen_arg_setup());
  concat(arg_setup, array->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TFArraySliceSymArg::gen_hard_constraint() const {
  assert(array != nullptr);
  std::vector<std::string> hard_ctrs;
  if (size_owned != nullptr)
    concat(hard_ctrs, size_owned->gen_hard_constraint());
  concat(hard_ctrs, array->gen_hard_constraint());
  return hard_ctrs;
}
std::vector<std::string> TFArraySliceSymArg::gen_soft_constraint() const {
  assert(array != nullptr);
  std::vector<std::string> soft_ctrs;
  if (size_owned != nullptr)
    concat(soft_ctrs, size_owned->gen_soft_constraint());
  concat(soft_ctrs, array->gen_soft_constraint());
  return soft_ctrs;
}
std::vector<std::string> TFArraySliceSymArg::gen_input_pass_condition() const {
  assert(array != nullptr);
  std::vector<std::string> ignore_conds;
  if (size_owned != nullptr)
    concat(ignore_conds, size_owned->gen_input_pass_condition());
  concat(ignore_conds, array->gen_input_pass_condition());
  return ignore_conds;
}
std::vector<std::string> TFArraySliceSymArg::gen_arg_initialization(
    GEN_MODE mode) const {
  assert(array != nullptr);
  std::vector<std::string> arg_initialization;
  if (size_owned != nullptr)
    concat(arg_initialization, size_owned->gen_arg_initialization(mode));
  concat(arg_initialization, array->gen_arg_initialization(mode));
  arg_initialization.push_back(type() + space + var() + assign +
                               initializer(mode) + semicolon + newline);
  tie_strs(arg_initialization);
  return arg_initialization;
}

void TFArraySliceSymArg::assign_concrete_enum_args(std::vector<long>& args) {
  if (size_owned != nullptr) size_owned->assign_concrete_enum_args(args);
  array->assign_concrete_enum_args(args);
}
void TFArraySliceSymArg::assign_concrete_int_args(std::vector<long>& args) {
  if (size_owned != nullptr) size_owned->assign_concrete_int_args(args);
  array->assign_concrete_int_args(args);
}

TFPartialTensorShapeSymArg::TFPartialTensorShapeSymArg(std::string name_)
    : TFSymArg(name_) {
  rank =
      std::make_unique<TFBoundedIntSymArg>(name + "_rank", 0, TF_MAX_RANK + 1);
  std::vector<std::unique_ptr<TFSymArg>> dims_vec;
  for (size_t i = 0; i < TF_MAX_RANK; i++)
    dims_vec.push_back(
        std::make_unique<TFIntSymArg>(name + "_" + std::to_string(i), 1));
  dims = std::make_unique<TFArraySymArg>(name + "_dims", std::move(dims_vec));
}

std::string TFPartialTensorShapeSymArg::type() const {
  return "PartialTensorShape";
}
std::string TFPartialTensorShapeSymArg::var() const { return name; }
std::string TFPartialTensorShapeSymArg::initializer(GEN_MODE mode) const {
  PDG_ASSERT(false);
}

void TFPartialTensorShapeSymArg::resolve_name_conflict(
    std::set<std::string>& names_seen) {
  TFSymArg::resolve_name_conflict(names_seen);
  rank->resolve_name_conflict(names_seen);
  dims->resolve_name_conflict(names_seen);
}

std::vector<std::string> TFPartialTensorShapeSymArg::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  concat(arg_setup, rank->gen_arg_setup());
  concat(arg_setup, dims->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TFPartialTensorShapeSymArg::gen_hard_constraint()
    const {
  std::vector<std::string> hard_ctrs;
  concat(hard_ctrs, rank->gen_hard_constraint());
  concat(hard_ctrs, dims->gen_soft_constraint());
  return hard_ctrs;
}
std::vector<std::string> TFPartialTensorShapeSymArg::gen_input_pass_condition()
    const {
  return {"is_too_big" + bracket(rank->expr(MODE_DRIVER) + comma +
                                 dims->initializer(MODE_DRIVER))};
}
std::vector<std::string> TFPartialTensorShapeSymArg::gen_arg_initialization(
    GEN_MODE mode) const {
  std::vector<std::string> arg_initialization;
  concat(arg_initialization, dims->gen_arg_initialization(mode));
  arg_initialization.push_back("PartialTensorShape" + space + name + semicolon);
  arg_initialization.push_back(
      "TF_CHECK_OK" +
      bracket("PartialTensorShape::MakePartialShape" +
              bracket(name + "_dims" + comma + rank->expr(mode) + comma + "&" +
                      name)) +
      semicolon + newline);
  tie_strs(arg_initialization);
  return arg_initialization;
}

void TFPartialTensorShapeSymArg::assign_concrete_enum_args(
    std::vector<long>& args) {
  rank->assign_concrete_enum_args(args);
  dims->assign_concrete_enum_args(args);
}
void TFPartialTensorShapeSymArg::assign_concrete_int_args(
    std::vector<long>& args) {
  rank->assign_concrete_int_args(args);
  dims->assign_concrete_int_args(args);
}

TFTensorSymArg::TFTensorSymArg(std::string name_, bool is_ref_,
                               std::unique_ptr<TFDtypeSymArg> dtype_owned_)
    : TFSymArg(name_), is_ref(is_ref_) {
  dtype_owned = std::move(dtype_owned_);

  rank =
      std::make_unique<TFBoundedIntSymArg>(name + "_rank", 0, TF_MAX_RANK + 1);
  std::vector<std::unique_ptr<TFSymArg>> dims_vec;
  for (size_t i = 0; i < TF_MAX_RANK; i++)
    dims_vec.push_back(
        std::make_unique<TFIntSymArg>(name + "_dim" + std::to_string(i), 1));
  dims = std::make_unique<TFArraySymArg>(name + "_dims", std::move(dims_vec));

  if (is_int_tensor()) {
    array_size = std::make_unique<TFBoundedIntSymArg>(name + "_array_size", 0,
                                                      TF_MAX_ARRAY_SIZE + 1);
    std::vector<std::unique_ptr<TFSymArg>> int_array_vec;
    for (size_t i = 0; i < TF_MAX_ARRAY_SIZE; i++)
      int_array_vec.push_back(
          std::make_unique<TFIntSymArg>(name + "_" + std::to_string(i), 1));
    array = std::make_unique<TFArraySymArg>(name + "_array",
                                            std::move(int_array_vec));
  }
}
TFTensorSymArg::TFTensorSymArg(std::string name_, bool is_ref_,
                               TFDtypeSymArg* dtype_dependent_)
    : TFSymArg(name_), is_ref(is_ref_) {
  dtype_dependent = dtype_dependent_;

  rank =
      std::make_unique<TFBoundedIntSymArg>(name + "_rank", 0, TF_MAX_RANK + 1);
  std::vector<std::unique_ptr<TFSymArg>> dims_vec;
  for (size_t i = 0; i < TF_MAX_RANK; i++)
    dims_vec.push_back(
        std::make_unique<TFIntSymArg>(name + "_dim" + std::to_string(i), 1));
  dims = std::make_unique<TFArraySymArg>(name + "_dims", std::move(dims_vec));

  if (is_int_tensor()) {
    array_size = std::make_unique<TFBoundedIntSymArg>(name + "_array_size", 0,
                                                      TF_MAX_ARRAY_SIZE + 1);
    std::vector<std::unique_ptr<TFSymArg>> int_array_vec;
    for (size_t i = 0; i < TF_MAX_ARRAY_SIZE; i++)
      int_array_vec.push_back(
          std::make_unique<TFIntSymArg>(name + "_" + std::to_string(i), 1));
    array = std::make_unique<TFArraySymArg>(name + "_array",
                                            std::move(int_array_vec));
  }
}

std::string TFTensorSymArg::type() const { return "Input"; }
std::string TFTensorSymArg::var() const { return name; }
std::string TFTensorSymArg::initializer(GEN_MODE mode) const {
  PDG_ASSERT(false);
}

void TFTensorSymArg::resolve_name_conflict(std::set<std::string>& names_seen) {
  TFSymArg::resolve_name_conflict(names_seen);
  if (dtype_owned != nullptr) dtype_owned->resolve_name_conflict(names_seen);
  rank->resolve_name_conflict(names_seen);
  dims->resolve_name_conflict(names_seen);
  if (is_int_tensor()) {
    array_size->resolve_name_conflict(names_seen);
    array->resolve_name_conflict(names_seen);
  }
}

std::vector<std::string> TFTensorSymArg::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  if (dtype_owned != nullptr) concat(arg_setup, dtype_owned->gen_arg_setup());
  concat(arg_setup, rank->gen_arg_setup());
  concat(arg_setup, dims->gen_arg_setup());
  if (is_int_tensor()) {
    concat(arg_setup, array_size->gen_arg_setup());
    concat(arg_setup, array->gen_arg_setup());
  }
  return arg_setup;
}
std::vector<std::string> TFTensorSymArg::gen_hard_constraint() const {
  std::vector<std::string> hard_ctrs;
  if (dtype_owned != nullptr)
    concat(hard_ctrs, dtype_owned->gen_hard_constraint());
  concat(hard_ctrs, rank->gen_hard_constraint());
  concat(hard_ctrs, dims->gen_soft_constraint());
  if (is_int_tensor()) {
    concat(hard_ctrs, array_size->gen_hard_constraint());
    concat(hard_ctrs, array->gen_hard_constraint());
  }
  return hard_ctrs;
}
std::vector<std::string> TFTensorSymArg::gen_soft_constraint() const {
  if (!is_int_tensor()) return {};

  std::vector<std::string> soft_ctrs;
  concat(soft_ctrs, array->gen_soft_constraint());
  return soft_ctrs;
}
std::vector<std::string> TFTensorSymArg::gen_input_pass_condition() const {
  return {"is_too_big" + bracket(rank->expr(MODE_DRIVER) + comma +
                                 dims->initializer(MODE_DRIVER))};
}
std::vector<std::string> TFTensorSymArg::gen_arg_initialization(
    GEN_MODE mode) const {
  std::vector<std::string> arg_initialization;

  if (dtype_owned != nullptr)
    concat(arg_initialization, dtype_owned->gen_arg_initialization(mode));

  std::string is_ref_str = is_ref ? "true" : "false";
  std::string dtype_expr = dtype_owned != nullptr ? dtype_owned->expr(mode)
                                                  : dtype_dependent->expr(mode);

  concat(arg_initialization, dims->gen_arg_initialization(mode));

  if (is_int_tensor()) {
    concat(arg_initialization, array->gen_arg_initialization(mode));

    std::string args = tf_scope_var + comma + dtype_expr + comma +
                       rank->expr(mode) + comma + dims->expr(mode) + comma +
                       array_size->expr(mode) + comma + array->expr(mode);

    arg_initialization.push_back(type() + space + var() + assign +
                                 "tf_int_tensor" + bracket(args) + semicolon +
                                 newline);
    tie_strs(arg_initialization);
    return arg_initialization;
  } else {
    std::string args = tf_scope_var + comma + is_ref_str + comma + dtype_expr +
                       comma + rank->expr(mode) + comma + dims->expr(mode);

    arg_initialization.push_back(type() + space + var() + assign + "tf_tensor" +
                                 bracket(args) + semicolon + newline);
    tie_strs(arg_initialization);
    return arg_initialization;
  }
}

void TFTensorSymArg::assign_concrete_enum_args(std::vector<long>& args) {
  if (dtype_owned != nullptr) dtype_owned->assign_concrete_enum_args(args);
  rank->assign_concrete_enum_args(args);
  dims->assign_concrete_enum_args(args);
  if (is_int_tensor()) {
    array_size->assign_concrete_enum_args(args);
    array->assign_concrete_enum_args(args);
  }
}
void TFTensorSymArg::assign_concrete_int_args(std::vector<long>& args) {
  if (dtype_owned != nullptr) dtype_owned->assign_concrete_int_args(args);
  rank->assign_concrete_int_args(args);
  dims->assign_concrete_int_args(args);
  if (is_int_tensor()) {
    array_size->assign_concrete_int_args(args);
    array->assign_concrete_int_args(args);
  }
}

bool TFTensorSymArg::is_int_tensor() const {
  if (is_ref) return false;

  if (dtype_owned != nullptr) {
    return dtype_owned->is_int_datatype();
  } else {
    return dtype_dependent->is_int_datatype();
  }
}

TFInputListSymArg::TFInputListSymArg(
    std::string name_, bool is_ref_,
    std::unique_ptr<TFDtypeSymArg> dtype_owned_, TFBoundedIntSymArg* size)
    : TFSymArg(name_), is_ref(is_ref_) {
  dtype_owned = std::move(dtype_owned_);
  std::vector<std::unique_ptr<TFSymArg>> inputs;
  for (size_t i = 0; i < size->max(); i++) {
    std::string input_name = name + "_" + std::to_string(i);
    inputs.push_back(std::make_unique<TFTensorSymArg>(input_name, is_ref,
                                                      dtype_owned.get()));
  }
  inputlist = std::make_unique<TFArraySliceSymArg>(name + "_array",
                                                   std::move(inputs), size);
}
TFInputListSymArg::TFInputListSymArg(std::string name_, bool is_ref_,
                                     TFDtypeSymArg* dtype,
                                     TFBoundedIntSymArg* size)
    : TFSymArg(name_), is_ref(is_ref_) {
  std::vector<std::unique_ptr<TFSymArg>> inputs;
  for (size_t i = 0; i < size->max(); i++) {
    std::string input_name = name + "_" + std::to_string(i);
    inputs.push_back(
        std::make_unique<TFTensorSymArg>(input_name, is_ref, dtype));
  }
  inputlist = std::make_unique<TFArraySliceSymArg>(name + "_array",
                                                   std::move(inputs), size);
}

std::string TFInputListSymArg::type() const { return "InputList"; }
std::string TFInputListSymArg::var() const { return name; }
std::string TFInputListSymArg::initializer(GEN_MODE mode) const {
  PDG_ASSERT(false);
}

void TFInputListSymArg::resolve_name_conflict(
    std::set<std::string>& names_seen) {
  if (dtype_owned != nullptr) dtype_owned->resolve_name_conflict(names_seen);
  inputlist->resolve_name_conflict(names_seen);
}

std::vector<std::string> TFInputListSymArg::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  if (dtype_owned != nullptr) concat(arg_setup, dtype_owned->gen_arg_setup());
  concat(arg_setup, inputlist->gen_arg_setup());
  return arg_setup;
}
std::vector<std::string> TFInputListSymArg::gen_hard_constraint() const {
  std::vector<std::string> hard_ctrs;
  if (dtype_owned != nullptr)
    concat(hard_ctrs, dtype_owned->gen_hard_constraint());
  concat(hard_ctrs, inputlist->gen_hard_constraint());
  return hard_ctrs;
}
std::vector<std::string> TFInputListSymArg::gen_soft_constraint() const {
  std::vector<std::string> soft_ctrs;
  if (dtype_owned != nullptr)
    concat(soft_ctrs, dtype_owned->gen_soft_constraint());
  concat(soft_ctrs, inputlist->gen_soft_constraint());
  return soft_ctrs;
}
std::vector<std::string> TFInputListSymArg::gen_input_pass_condition() const {
  std::vector<std::string> ignore_conds;
  if (dtype_owned != nullptr)
    concat(ignore_conds, dtype_owned->gen_input_pass_condition());
  concat(ignore_conds, inputlist->gen_input_pass_condition());
  return ignore_conds;
}
std::vector<std::string> TFInputListSymArg::gen_arg_initialization(
    GEN_MODE mode) const {
  std::vector<std::string> arg_initialization;
  if (dtype_owned != nullptr)
    concat(arg_initialization, dtype_owned->gen_arg_initialization(mode));
  concat(arg_initialization, inputlist->gen_arg_initialization(mode));
  arg_initialization.push_back(type() + space + var() + assign +
                               inputlist->expr(mode) + semicolon + newline);
  tie_strs(arg_initialization);
  return arg_initialization;
}

void TFInputListSymArg::assign_concrete_enum_args(std::vector<long>& args) {
  if (dtype_owned != nullptr) dtype_owned->assign_concrete_enum_args(args);
  inputlist->assign_concrete_enum_args(args);
}
void TFInputListSymArg::assign_concrete_int_args(std::vector<long>& args) {
  if (dtype_owned != nullptr) dtype_owned->assign_concrete_int_args(args);
  inputlist->assign_concrete_int_args(args);
}

TFAPIAttrsSymArg::TFAPIAttrsSymArg(
    std::string name_, std::string api_attrs_class_name_,
    std::vector<std::pair<std::string, std::unique_ptr<TFSymArg>>> setters_)
    : TFSymArg(name_) {
  api_attrs_class_name = api_attrs_class_name_;
  setters = std::move(setters_);
}

std::string TFAPIAttrsSymArg::type() const { return api_attrs_class_name; }
std::string TFAPIAttrsSymArg::var() const { return name; }
std::string TFAPIAttrsSymArg::initializer(GEN_MODE mode) const {
  PDG_ASSERT(false);
}

void TFAPIAttrsSymArg::resolve_name_conflict(
    std::set<std::string>& names_seen) {
  TFSymArg::resolve_name_conflict(names_seen);
  for (auto& t : setters) {
    auto& setter_symarg = std::get<1>(t);
    setter_symarg->resolve_name_conflict(names_seen);
  }
}

std::vector<std::string> TFAPIAttrsSymArg::gen_arg_setup() const {
  std::vector<std::string> arg_setup;
  for (auto& t : setters) {
    auto& setter_symarg = std::get<1>(t);
    concat(arg_setup, setter_symarg->gen_arg_setup());
  }
  return arg_setup;
}
std::vector<std::string> TFAPIAttrsSymArg::gen_hard_constraint() const {
  std::vector<std::string> hard_ctrs;
  for (auto& t : setters) {
    auto& setter_symarg = std::get<1>(t);
    concat(hard_ctrs, setter_symarg->gen_hard_constraint());
  }
  return hard_ctrs;
}
std::vector<std::string> TFAPIAttrsSymArg::gen_soft_constraint() const {
  std::vector<std::string> soft_ctrs;
  for (auto& t : setters) {
    auto& setter_symarg = std::get<1>(t);
    concat(soft_ctrs, setter_symarg->gen_soft_constraint());
  }
  return soft_ctrs;
}
std::vector<std::string> TFAPIAttrsSymArg::gen_input_pass_condition() const {
  std::vector<std::string> ignore_conds;
  for (auto& t : setters) {
    auto& setter_symarg = std::get<1>(t);
    concat(ignore_conds, setter_symarg->gen_input_pass_condition());
  }
  return ignore_conds;
}
std::vector<std::string> TFAPIAttrsSymArg::gen_arg_initialization(
    GEN_MODE mode) const {
  std::vector<std::string> arg_initialization;
  for (auto& t : setters) {
    auto& setter_symarg = std::get<1>(t);
    concat(arg_initialization, setter_symarg->gen_arg_initialization(mode));
  }
  concat(arg_initialization, gen_api_attrs_init(mode));
  if (!arg_initialization.empty())
    arg_initialization.back() = arg_initialization.back() + newline;
  return arg_initialization;
}

void TFAPIAttrsSymArg::assign_concrete_enum_args(std::vector<long>& args) {
  for (auto& t : setters) {
    auto& setter_symarg = std::get<1>(t);
    setter_symarg->assign_concrete_enum_args(args);
  }
}
void TFAPIAttrsSymArg::assign_concrete_int_args(std::vector<long>& args) {
  for (auto& t : setters) {
    auto& setter_symarg = std::get<1>(t);
    setter_symarg->assign_concrete_int_args(args);
  }
}

std::vector<std::string> TFAPIAttrsSymArg::gen_api_attrs_init(
    GEN_MODE mode) const {
  std::vector<std::string> api_attrs_init;

  api_attrs_init.push_back("auto " + name + assign);
  api_attrs_init.push_back("  " + api_attrs_class_name + "()");
  for (auto& t : setters) {
    auto& setter_name = std::get<0>(t);
    auto& setter_symarg = std::get<1>(t);
    api_attrs_init.push_back("    ." + setter_name +
                             bracket(setter_symarg->expr(mode)));
  }
  api_attrs_init.back() = api_attrs_init.back() + semicolon;
  return api_attrs_init;
}
