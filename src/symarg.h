#ifndef PATHFINDER_DRIVER_GENERATOR_SYMARG
#define PATHFINDER_DRIVER_GENERATOR_SYMARG

#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "utils.h"

const std::string symbolic_int_var = "sym_int_arg";
const std::string callback_input_var = "x";
const std::string tf_scope_var = "scope";

enum GEN_MODE {
  MODE_DRIVER,
  MODE_POV,
};

class SymArg {
 public:
  SymArg(std::string name_);
  virtual ~SymArg() = default;

  virtual std::string type() const = 0;
  virtual std::string var() const;
  virtual std::string initializer(GEN_MODE mode) const = 0;
  std::string expr(GEN_MODE mode) const;

  virtual void resolve_name_conflict(std::set<std::string>& names_seen);

  virtual std::vector<std::string> gen_arg_setup() const;
  virtual std::vector<std::string> gen_hard_constraint() const;
  virtual std::vector<std::string> gen_soft_constraint() const;
  virtual std::vector<std::string> gen_input_pass_condition() const;
  virtual std::vector<std::string> gen_arg_initialization(GEN_MODE mode) const;

  virtual void assign_concrete_enum_args(std::vector<long>& args);
  virtual void assign_concrete_int_args(std::vector<long>& args);

  virtual bool stable() const;
  std::string get_name() const;

 protected:
  std::string name;
};

class NumSymArg : public SymArg {
 public:
  NumSymArg(std::string name_);

  virtual std::string type() const override = 0;
  virtual std::string initializer(GEN_MODE mode) const override = 0;

  virtual std::vector<std::string> gen_arg_setup() const override;

  virtual void assign_concrete_int_args(std::vector<long>& args) override;

 protected:
  std::optional<long> concrete_val;
};

class BoundedSymArg : public SymArg {
 public:
  BoundedSymArg(std::string name_,
                const std::vector<std::string>& value_names_);
  BoundedSymArg(std::string name_, int start_, size_t size_);
  BoundedSymArg(std::string name_, const std::string& value_list_var_);

  virtual std::string type() const override = 0;
  virtual std::string initializer(GEN_MODE mode) const override = 0;

  virtual std::vector<std::string> gen_arg_setup() const override;
  virtual std::vector<std::string> gen_hard_constraint() const override;

  virtual void assign_concrete_enum_args(std::vector<long>& args) override;

 protected:
  std::vector<std::string> value_names;
  std::optional<std::pair<int, size_t>> start_size_pair;
  std::string value_list_var;

  std::optional<long> concrete_val;
};

std::string setup_var(std::string param_name);
std::string callback_var(std::string param_name);
std::vector<std::string> get_names(
    const std::vector<std::unique_ptr<SymArg>>& symargs);

template <typename T>
std::string to_init_list(const std::vector<std::unique_ptr<T>>& params,
                         GEN_MODE mode) {
  std::string str;
  for (size_t i = 0; i < params.size(); i++) {
    str += params[i]->expr(mode);
    if (i != params.size() - 1) str += comma;
  }
  return curly(str);
}

#endif
