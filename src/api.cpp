#include "api.h"

API::API(std::string api_name_, std::set<std::string> reserved_names_)
    : api_name(api_name_), reserved_names(reserved_names_) {}
void API::resolve_name_conflict() {
  std::set<std::string> names_seen = reserved_names;
  for (auto& symarg : symargs) symarg->resolve_name_conflict(names_seen);
}
std::string API::get_name() const { return api_name; }
const std::vector<std::unique_ptr<SymArg>>& API::get_symargs() const {
  return symargs;
}
void API::assign(std::vector<long> args) {
  assign_concrete_enum_args(args);
  assign_concrete_int_args(args);
  is_pov_mode = true;
}
void API::assign_concrete_enum_args(std::vector<long>& args) {
  for (auto& symarg : symargs) symarg->assign_concrete_enum_args(args);
}
void API::assign_concrete_int_args(std::vector<long>& args) {
  for (auto& symarg : symargs) symarg->assign_concrete_int_args(args);
}
