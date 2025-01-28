#ifndef PATHFINDER_DRIVER_GENERATOR_GENERATOR
#define PATHFINDER_DRIVER_GENERATOR_GENERATOR

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "api.h"
#include "utils.h"

template <typename T>
class Generator {
 public:
  Generator();
  virtual std::string gen(T* api_) = 0;

 protected:
  T* api;
};

template <typename T>
class FuzzDriverGenerator : public Generator<T> {
 public:
  FuzzDriverGenerator();
  virtual std::string gen(T* api_) override;

 protected:
  std::vector<std::string> arg_setup_code() const;
  std::vector<std::string> hard_constraint_code() const;
  std::vector<std::string> soft_constraint_code() const;
  std::vector<std::string> input_pass_condition_code() const;
  std::vector<std::string> arg_initialization_code() const;

  T* api;

 private:
  virtual std::vector<std::string> header() const = 0;
  virtual std::vector<std::string> setup() const = 0;
  virtual std::vector<std::string> callback() const = 0;
  virtual std::vector<std::string> footer() const = 0;
};

template <typename T>
Generator<T>::Generator() {}

template <typename T>
FuzzDriverGenerator<T>::FuzzDriverGenerator() {}

template <typename T>
std::string FuzzDriverGenerator<T>::gen(T* api_) {
  api = api_;
  std::vector<std::string> lines;
  concat(lines, header());
  concat(lines, setup());
  concat(lines, callback());
  concat(lines, footer());
  return join_strs(lines, newline);
}

template <typename T>
std::vector<std::string> FuzzDriverGenerator<T>::arg_setup_code() const {
  std::vector<std::string> arg_setup;
  for (auto& symarg : api->get_symargs())
    concat(arg_setup, symarg->gen_arg_setup());
  return arg_setup;
}

template <typename T>
std::vector<std::string> FuzzDriverGenerator<T>::hard_constraint_code() const {
  std::vector<std::string> hard_constraint;
  for (auto& symarg : api->get_symargs())
    concat(hard_constraint, "  ", symarg->gen_hard_constraint(), comma);

  if (hard_constraint.empty()) return {};

  hard_constraint.insert(hard_constraint.begin(),
                         "PathFinderAddHardConstraint({");
  hard_constraint.push_back("});");
  return hard_constraint;
}

template <typename T>
std::vector<std::string> FuzzDriverGenerator<T>::soft_constraint_code() const {
  std::vector<std::string> soft_constraint;
  for (auto& symarg : api->get_symargs())
    concat(soft_constraint, "  ", symarg->gen_soft_constraint(), comma);

  if (soft_constraint.empty()) return {};

  soft_constraint.insert(soft_constraint.begin(),
                         "PathFinderAddSoftConstraint({");
  soft_constraint.push_back("});");
  return soft_constraint;
}

template <typename T>
std::vector<std::string> FuzzDriverGenerator<T>::input_pass_condition_code()
    const {
  std::vector<std::string> input_pass_condition;
  for (auto& symarg : api->get_symargs())
    concat(input_pass_condition, "PathFinderPassIf(",
           symarg->gen_input_pass_condition(), ");");
  return input_pass_condition;
}

template <typename T>
std::vector<std::string> FuzzDriverGenerator<T>::arg_initialization_code()
    const {
  std::vector<std::string> arg_initialization;
  for (auto& symarg : api->get_symargs())
    concat(arg_initialization, symarg->gen_arg_initialization(MODE_DRIVER));
  return arg_initialization;
}

template <typename T>
class POVGenerator : public Generator<T> {
 public:
  POVGenerator();
  virtual std::string gen(T* api_) override;

 protected:
  std::vector<std::string> arg_initialization_code() const;

  T* api;

 private:
  virtual std::vector<std::string> header() const = 0;
  virtual std::vector<std::string> callback() const = 0;
};

template <typename T>
POVGenerator<T>::POVGenerator() {}

template <typename T>
std::string POVGenerator<T>::gen(T* api_) {
  api = api_;
  std::vector<std::string> lines;
  concat(lines, header());
  concat(lines, callback());
  return join_strs(lines, newline);
}

template <typename T>
std::vector<std::string> POVGenerator<T>::arg_initialization_code() const {
  std::vector<std::string> arg_initialization;
  for (auto& symarg : api->get_symargs())
    concat(arg_initialization, symarg->gen_arg_initialization(MODE_POV));
  return arg_initialization;
}

#endif
