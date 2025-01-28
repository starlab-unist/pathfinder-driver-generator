#ifndef PATHFINDER_DRIVER_GENERATOR_API
#define PATHFINDER_DRIVER_GENERATOR_API

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "symarg.h"
#include "utils.h"

class API {
 public:
  API(std::string api_name_, std::set<std::string> reserved_names);
  void resolve_name_conflict();

  std::string get_name() const;
  const std::vector<std::unique_ptr<SymArg>>& get_symargs() const;

  void assign(std::vector<long> args);

 protected:
  std::vector<std::unique_ptr<SymArg>> symargs;
  bool is_pov_mode = false;

 private:
  void assign_concrete_enum_args(std::vector<long>& args);
  void assign_concrete_int_args(std::vector<long>& args);

  std::string api_name;
  std::set<std::string> reserved_names;
};

#endif
