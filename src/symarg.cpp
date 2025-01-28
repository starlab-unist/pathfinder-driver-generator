#include "symarg.h"

#include "options.h"

SymArg::SymArg(std::string name_) : name(name_) {}
std::string SymArg::var() const { return ""; }
std::string SymArg::expr(GEN_MODE mode) const {
  return var() != "" ? var() : initializer(mode);
};
void SymArg::resolve_name_conflict(std::set<std::string>& names_seen) {
  name = unique_name(name, names_seen);
}
std::vector<std::string> SymArg::gen_arg_setup() const { return {}; }
std::vector<std::string> SymArg::gen_hard_constraint() const { return {}; }
std::vector<std::string> SymArg::gen_soft_constraint() const { return {}; }
std::vector<std::string> SymArg::gen_input_pass_condition() const { return {}; }
std::vector<std::string> SymArg::gen_arg_initialization(GEN_MODE mode) const {
  return {};
}
void SymArg::assign_concrete_enum_args(std::vector<long>& args) {}
void SymArg::assign_concrete_int_args(std::vector<long>& args) {}
bool SymArg::stable() const { return true; }
std::string SymArg::get_name() const { return name; }

NumSymArg::NumSymArg(std::string name_) : SymArg(name_) {}
std::vector<std::string> NumSymArg::gen_arg_setup() const {
  return {"PathFinderIntArg" + bracket(quoted(name)) + semicolon};
}
void NumSymArg::assign_concrete_int_args(std::vector<long>& args) {
  concrete_val = head(args);
  args = tail(args);
}

BoundedSymArg::BoundedSymArg(std::string name_,
                             const std::vector<std::string>& value_names_)
    : SymArg(name_) {
  for (auto value_name_ : value_names_) value_names.push_back(value_name_);
}
BoundedSymArg::BoundedSymArg(std::string name_, int start_, size_t size_)
    : SymArg(name_) {
  start_size_pair = {start_, size_};
}
BoundedSymArg::BoundedSymArg(std::string name_,
                             const std::string& value_list_var_)
    : SymArg(name_) {
  value_list_var = value_list_var_;
}
std::vector<std::string> BoundedSymArg::gen_arg_setup() const {
  if (WO_STAGED) {
    return {"PathFinderIntArg" + bracket(quoted(name)) + semicolon};
  } else {
    std::string setup_args;
    if (!value_names.empty()) {
      setup_args = quoted(name) + comma + curly(join_quoted_strs(value_names));
    } else if (start_size_pair.has_value()) {
      setup_args = quoted(name) + comma +
                   std::to_string(start_size_pair.value().first) + comma +
                   std::to_string(start_size_pair.value().second);
    } else {
      setup_args = quoted(name) + comma + value_list_var;
    }

    return {"PathFinderEnumArg" + bracket(setup_args) + semicolon};
  }
}
std::vector<std::string> BoundedSymArg::gen_hard_constraint() const {
  if (WO_STAGED) {
    std::string min;
    std::string max;
    if (!value_names.empty()) {
      min = std::to_string(0);
      max = std::to_string(value_names.size() - 1);
    } else if (start_size_pair.has_value()) {
      int start = start_size_pair.value().first;
      size_t size = start_size_pair.value().second;
      min = std::to_string(start);
      max = std::to_string(start + size - 1);
    } else {
      min = std::to_string(0);
      max = value_list_var + " - 1";
    }

    return {min + lte + setup_var(name) + comma + setup_var(name) + lte + max};
  } else {
    return {};
  }
}
void BoundedSymArg::assign_concrete_enum_args(std::vector<long>& args) {
  concrete_val = head(args);
  args = tail(args);
}

std::string setup_var(std::string symarg_name) {
  return symbolic_int_var + sq_quoted(symarg_name);
}
std::string callback_var(std::string symarg_name) {
  return callback_input_var + sq_quoted(symarg_name);
}
std::vector<std::string> get_names(
    const std::vector<std::unique_ptr<SymArg>>& symargs) {
  std::vector<std::string> names;
  for (auto& symarg : symargs) names.push_back(symarg->get_name());
  return names;
}
