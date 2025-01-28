#ifndef PATHFINDER_DRIVER_GENERATOR_TF_SYMARG
#define PATHFINDER_DRIVER_GENERATOR_TF_SYMARG

#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "symarg.h"
#include "tf_datatype.h"

extern const size_t TF_MAX_RANK;
extern const size_t TF_MAX_ARRAY_SIZE;

typedef SymArg TFSymArg;
typedef NumSymArg TFNumSymArg;
typedef BoundedSymArg TFBoundedSymArg;

class TFScopeSymArg : public TFSymArg {
 public:
  TFScopeSymArg();

  virtual std::string type() const override;
  virtual std::string var() const override;
  virtual std::string initializer(GEN_MODE mode) const override;

  virtual void resolve_name_conflict(
      std::set<std::string>& names_seen) override;

  virtual std::vector<std::string> gen_arg_initialization(
      GEN_MODE mode) const override;
};

class TFIntSymArg : public TFNumSymArg {
 public:
  TFIntSymArg(std::string name_, int min_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;

  virtual std::vector<std::string> gen_soft_constraint() const override;

 private:
  int min;
};

class TFBoundedIntSymArg : public TFBoundedSymArg {
 public:
  TFBoundedIntSymArg(std::string name_, int start_, size_t size_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;

  int min() const;
  int max() const;
};

class TFBoolSymArg : public TFBoundedSymArg {
 public:
  TFBoolSymArg(std::string name_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;
};

class TFFloatSymArg : public TFBoundedSymArg {
 public:
  TFFloatSymArg(std::string name_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;
};

class TFDoubleSymArg : public TFBoundedSymArg {
 public:
  TFDoubleSymArg(std::string name_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;
};

class TFDtypeSymArg : public TFBoundedSymArg {
 public:
  TFDtypeSymArg(std::string name_, const std::vector<std::string>& allowed_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;

  virtual std::vector<std::string> gen_arg_initialization(
      GEN_MODE mode) const override;

  bool is_int_datatype() const;

 private:
  std::string allowed_list_name() const;
};

class TFStringSymArg : public TFBoundedSymArg {
 public:
  TFStringSymArg(std::string name_, const std::vector<std::string>& allowed_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;

  virtual std::vector<std::string> gen_arg_initialization(
      GEN_MODE mode) const override;

  void to_tstring();

 private:
  std::string allowed_list_name() const;

  bool tstring = false;
};

class TFArraySymArg : public TFSymArg {
 public:
  TFArraySymArg(std::string name_,
                std::vector<std::unique_ptr<TFSymArg>> symargs_);

  virtual std::string type() const override;
  virtual std::string var() const override;
  virtual std::string initializer(GEN_MODE mode) const override;

  virtual void resolve_name_conflict(
      std::set<std::string>& names_seen) override;

  virtual std::vector<std::string> gen_arg_setup() const override;
  virtual std::vector<std::string> gen_hard_constraint() const override;
  virtual std::vector<std::string> gen_soft_constraint() const override;
  virtual std::vector<std::string> gen_input_pass_condition() const override;
  virtual std::vector<std::string> gen_arg_initialization(
      GEN_MODE mode) const override;

  virtual void assign_concrete_enum_args(std::vector<long>& args) override;
  virtual void assign_concrete_int_args(std::vector<long>& args) override;

  std::string base_type() const;

 private:
  std::vector<std::unique_ptr<TFSymArg>> symargs;
};

class TFArraySliceSymArg : public TFSymArg {
 public:
  TFArraySliceSymArg(std::string name_,
                     std::vector<std::unique_ptr<TFSymArg>> symargs_,
                     size_t size_min, size_t size_max);
  TFArraySliceSymArg(std::string name_,
                     std::vector<std::unique_ptr<TFSymArg>> symargs_,
                     TFBoundedIntSymArg* size_dependent_);

  virtual std::string type() const override;
  virtual std::string var() const override;
  virtual std::string initializer(GEN_MODE mode) const override;

  virtual void resolve_name_conflict(
      std::set<std::string>& names_seen) override;

  virtual std::vector<std::string> gen_arg_setup() const override;
  virtual std::vector<std::string> gen_hard_constraint() const override;
  virtual std::vector<std::string> gen_soft_constraint() const override;
  virtual std::vector<std::string> gen_input_pass_condition() const override;
  virtual std::vector<std::string> gen_arg_initialization(
      GEN_MODE mode) const override;

  virtual void assign_concrete_enum_args(std::vector<long>& args) override;
  virtual void assign_concrete_int_args(std::vector<long>& args) override;

 private:
  std::optional<size_t> const_size;
  TFBoundedIntSymArg* size_dependent;
  std::unique_ptr<TFBoundedIntSymArg> size_owned;
  std::unique_ptr<TFArraySymArg> array;
};

class TFPartialTensorShapeSymArg : public TFSymArg {
 public:
  TFPartialTensorShapeSymArg(std::string name_);

  virtual std::string type() const override;
  virtual std::string var() const override;
  virtual std::string initializer(GEN_MODE mode) const override;

  virtual void resolve_name_conflict(
      std::set<std::string>& names_seen) override;

  virtual std::vector<std::string> gen_arg_setup() const override;
  virtual std::vector<std::string> gen_hard_constraint() const override;
  virtual std::vector<std::string> gen_input_pass_condition() const override;
  virtual std::vector<std::string> gen_arg_initialization(
      GEN_MODE mode) const override;

  virtual void assign_concrete_enum_args(std::vector<long>& args) override;
  virtual void assign_concrete_int_args(std::vector<long>& args) override;

 private:
  std::unique_ptr<TFBoundedIntSymArg> rank;
  std::unique_ptr<TFArraySymArg> dims;
};

class TFTensorSymArg : public TFSymArg {
 public:
  TFTensorSymArg(std::string name_, bool is_ref_,
                 std::unique_ptr<TFDtypeSymArg> dtype_owned_);
  TFTensorSymArg(std::string name_, bool is_ref_,
                 TFDtypeSymArg* dtype_dependent_);

  virtual std::string type() const override;
  virtual std::string var() const override;
  virtual std::string initializer(GEN_MODE mode) const override;

  virtual void resolve_name_conflict(
      std::set<std::string>& names_seen) override;

  virtual std::vector<std::string> gen_arg_setup() const override;
  virtual std::vector<std::string> gen_hard_constraint() const override;
  virtual std::vector<std::string> gen_soft_constraint() const override;
  virtual std::vector<std::string> gen_input_pass_condition() const override;
  virtual std::vector<std::string> gen_arg_initialization(
      GEN_MODE mode) const override;

  virtual void assign_concrete_enum_args(std::vector<long>& args) override;
  virtual void assign_concrete_int_args(std::vector<long>& args) override;

 private:
  bool is_int_tensor() const;

  bool is_ref;
  std::unique_ptr<TFDtypeSymArg> dtype_owned;
  TFDtypeSymArg* dtype_dependent;
  std::unique_ptr<TFBoundedIntSymArg> rank;
  std::unique_ptr<TFArraySymArg> dims;
  std::unique_ptr<TFBoundedIntSymArg> array_size;
  std::unique_ptr<TFArraySymArg> array;
};

class TFInputListSymArg : public TFSymArg {
 public:
  TFInputListSymArg(std::string name_, bool is_ref_,
                    std::unique_ptr<TFDtypeSymArg> dtype_owned_,
                    TFBoundedIntSymArg* size_);
  TFInputListSymArg(std::string name_, bool is_ref_, TFDtypeSymArg* dtype_,
                    TFBoundedIntSymArg* size_);

  virtual std::string type() const override;
  virtual std::string var() const override;
  virtual std::string initializer(GEN_MODE mode) const override;

  virtual void resolve_name_conflict(
      std::set<std::string>& names_seen) override;

  virtual std::vector<std::string> gen_arg_setup() const override;
  virtual std::vector<std::string> gen_hard_constraint() const override;
  virtual std::vector<std::string> gen_soft_constraint() const override;
  virtual std::vector<std::string> gen_input_pass_condition() const override;
  virtual std::vector<std::string> gen_arg_initialization(
      GEN_MODE mode) const override;

  virtual void assign_concrete_enum_args(std::vector<long>& args) override;
  virtual void assign_concrete_int_args(std::vector<long>& args) override;

 private:
  bool is_ref;
  std::unique_ptr<TFDtypeSymArg> dtype_owned;
  std::unique_ptr<TFArraySliceSymArg> inputlist;
};

class TFAPIAttrsSymArg : public TFSymArg {
 public:
  TFAPIAttrsSymArg(
      std::string name_, std::string api_attrs_class_name_,
      std::vector<std::pair<std::string, std::unique_ptr<TFSymArg>>> setters_);

  virtual std::string type() const override;
  virtual std::string var() const override;
  virtual std::string initializer(GEN_MODE mode) const override;

  virtual void resolve_name_conflict(
      std::set<std::string>& names_seen) override;

  virtual std::vector<std::string> gen_arg_setup() const override;
  virtual std::vector<std::string> gen_hard_constraint() const override;
  virtual std::vector<std::string> gen_soft_constraint() const override;
  virtual std::vector<std::string> gen_input_pass_condition() const override;
  virtual std::vector<std::string> gen_arg_initialization(
      GEN_MODE mode) const override;

  virtual void assign_concrete_enum_args(std::vector<long>& args) override;
  virtual void assign_concrete_int_args(std::vector<long>& args) override;

 private:
  std::string api_attrs_class_name;
  std::vector<std::pair<std::string, std::unique_ptr<TFSymArg>>> setters;

  std::vector<std::string> gen_api_attrs_init(GEN_MODE mode) const;
};

#endif
