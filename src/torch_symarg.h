#ifndef PATHFINDER_DRIVER_GENERATOR_TORCH_SYMARG
#define PATHFINDER_DRIVER_GENERATOR_TORCH_SYMARG

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "symarg.h"
#include "utils.h"

extern const size_t TORCH_MAX_RANK;
extern const size_t TORCH_MAX_VECTOR_SIZE;
extern const size_t TORCH_MAX_ARRAYREF_SIZE;

typedef SymArg TorchSymArg;
typedef NumSymArg TorchNumSymArg;
typedef BoundedSymArg TorchBoundedSymArg;

class TorchIntSymArg : public TorchNumSymArg {
 public:
  TorchIntSymArg(std::string name_, std::string specifier_);
  void set_default(int value);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;

  virtual std::vector<std::string> gen_soft_constraint() const override;

 private:
  std::string specifier;
  std::optional<int> default_value = std::nullopt;
};

class TorchSymIntSymArg : public TorchSymArg {
 public:
  TorchSymIntSymArg(std::string name_);
  void set_default(int value);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;

  virtual std::vector<std::string> gen_arg_setup() const override;
  virtual std::vector<std::string> gen_soft_constraint() const override;

 private:
  std::unique_ptr<TorchIntSymArg> intsymarg;
};

class TorchUnsignedIntSymArg : public TorchNumSymArg {
 public:
  TorchUnsignedIntSymArg(std::string name_);
  void set_default(unsigned int value);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;

  virtual std::vector<std::string> gen_hard_constraint() const override;
  virtual std::vector<std::string> gen_soft_constraint() const override;

 private:
  std::optional<unsigned int> default_value = std::nullopt;
};

class TorchNullSymArg : public TorchBoundedSymArg {
 public:
  TorchNullSymArg(std::string name_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;
};

class TorchBoundedIntSymArg : public TorchBoundedSymArg {
 public:
  TorchBoundedIntSymArg(std::string name_, size_t size_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;
};

class TorchBoolSymArg : public TorchBoundedSymArg {
 public:
  TorchBoolSymArg(std::string name_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;
};

class TorchStringSymArg : public TorchBoundedSymArg {
 public:
  TorchStringSymArg(std::string name_, bool is_view_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;

 private:
  bool is_view;
};

class TorchBFloatSymArg : public TorchBoundedSymArg {
 public:
  TorchBFloatSymArg(std::string name_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;
};

class TorchHalfSymArg : public TorchBoundedSymArg {
 public:
  TorchHalfSymArg(std::string name_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;
};

class TorchFloatSymArg : public TorchBoundedSymArg {
 public:
  TorchFloatSymArg(std::string name_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;
};

class TorchDoubleSymArg : public TorchBoundedSymArg {
 public:
  TorchDoubleSymArg(std::string name_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;
};

class TorchMemoryFormatSymArg : public TorchBoundedSymArg {
 public:
  TorchMemoryFormatSymArg(std::string name_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;
};

class TorchLayoutSymArg : public TorchBoundedSymArg {
 public:
  TorchLayoutSymArg(std::string name_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;
};

class TorchDeviceSymArg : public TorchBoundedSymArg {
 public:
  TorchDeviceSymArg(std::string name_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;
};

class TorchDtypeSymArg : public TorchBoundedSymArg {
 public:
  TorchDtypeSymArg(std::string name_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;
};

class TorchScalarDtypeSymArg : public TorchBoundedSymArg {
 public:
  TorchScalarDtypeSymArg(std::string name_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;
};

class TorchVariantSymArg : public TorchBoundedSymArg {
 public:
  TorchVariantSymArg(std::string name_,
                     std::vector<std::unique_ptr<TorchSymArg>> symargs_);

  virtual std::string type() const override;
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
  std::vector<std::unique_ptr<TorchSymArg>> symargs;

  std::vector<std::string> gen_typedef() const;
  std::vector<std::string> gen_vector(GEN_MODE mode) const;
};

class TorchEnumSymArg : public TorchSymArg {
 public:
  TorchEnumSymArg(std::string name_, std::string enum_name_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;

 private:
  std::string enum_name;
};

class TorchUnfixedArraySymArg : public TorchSymArg {
 public:
  TorchUnfixedArraySymArg(std::string name_,
                          std::vector<std::unique_ptr<TorchSymArg>> symargs_);
  TorchUnfixedArraySymArg(std::string name_, std::string base_type_str_);

  virtual std::string type() const override = 0;
  virtual std::string var() const override;
  virtual std::string initializer(GEN_MODE mode) const override = 0;

  virtual void resolve_name_conflict(
      std::set<std::string>& names_seen) override;

  virtual std::vector<std::string> gen_arg_setup() const override;
  virtual std::vector<std::string> gen_hard_constraint() const override;
  virtual std::vector<std::string> gen_soft_constraint() const override;
  virtual std::vector<std::string> gen_arg_initialization(
      GEN_MODE mode) const override;

  virtual void assign_concrete_enum_args(std::vector<long>& args) override;
  virtual void assign_concrete_int_args(std::vector<long>& args) override;

  virtual bool stable() const override;
  std::string base_type() const;

 protected:
  std::unique_ptr<TorchBoundedIntSymArg> size;
  std::vector<std::unique_ptr<TorchSymArg>> symargs;
  std::string base_type_str;
};

class TorchVectorSymArg : public TorchUnfixedArraySymArg {
 public:
  TorchVectorSymArg(std::string name_,
                    std::vector<std::unique_ptr<TorchSymArg>> symargs_);
  TorchVectorSymArg(std::string name_, std::string base_type_str_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;

 private:
  std::string base_type_str;
};

class TorchArrayRefSymArg : public TorchSymArg {
 public:
  TorchArrayRefSymArg(std::string name_,
                      std::vector<std::unique_ptr<TorchSymArg>> symargs_);
  TorchArrayRefSymArg(std::string name_, std::string base_type_str_);

  virtual std::string type() const override;
  virtual std::string var() const override;
  virtual std::string initializer(GEN_MODE mode) const override;

  virtual void resolve_name_conflict(
      std::set<std::string>& names_seen) override;

  virtual std::vector<std::string> gen_arg_setup() const override;
  virtual std::vector<std::string> gen_hard_constraint() const override;
  virtual std::vector<std::string> gen_soft_constraint() const override;
  virtual std::vector<std::string> gen_arg_initialization(
      GEN_MODE mode) const override;

  virtual void assign_concrete_enum_args(std::vector<long>& args) override;
  virtual void assign_concrete_int_args(std::vector<long>& args) override;

 private:
  std::unique_ptr<TorchVectorSymArg> vec;
};

class TorchFixedArraySymArg : public TorchSymArg {
 public:
  TorchFixedArraySymArg(std::string name_, size_t size_,
                        std::vector<std::unique_ptr<TorchSymArg>> symargs_);

  virtual std::string type() const override = 0;
  virtual std::string var() const override;
  virtual std::string initializer(GEN_MODE mode) const override = 0;

  virtual void resolve_name_conflict(
      std::set<std::string>& names_seen) override;

  virtual std::vector<std::string> gen_arg_setup() const override;
  virtual std::vector<std::string> gen_hard_constraint() const override;
  virtual std::vector<std::string> gen_soft_constraint() const override;
  virtual std::vector<std::string> gen_arg_initialization(
      GEN_MODE mode) const override;

  virtual void assign_concrete_enum_args(std::vector<long>& args) override;
  virtual void assign_concrete_int_args(std::vector<long>& args) override;

 protected:
  size_t size;
  std::vector<std::unique_ptr<TorchSymArg>> symargs;
};

class TorchExpandingArraySymArg : public TorchFixedArraySymArg {
 public:
  TorchExpandingArraySymArg(std::string name_, size_t size_,
                            std::vector<std::unique_ptr<TorchSymArg>> symargs_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;
};

class TorchExpandingArrayWithOptionalElemSymArg : public TorchFixedArraySymArg {
 public:
  TorchExpandingArrayWithOptionalElemSymArg(
      std::string name_, size_t size_,
      std::vector<std::unique_ptr<TorchSymArg>> symargs_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;

 private:
  std::string base_type() const;
};

class TorchTupleSymArg : public TorchFixedArraySymArg {
 public:
  TorchTupleSymArg(std::string name_, size_t size_,
                   std::vector<std::unique_ptr<TorchSymArg>> symargs_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;
};

class TorchPairSymArg : public TorchFixedArraySymArg {
 public:
  TorchPairSymArg(std::string name_,
                  std::vector<std::unique_ptr<TorchSymArg>> symargs_);

  virtual std::string type() const override;
  virtual std::string initializer(GEN_MODE mode) const override;
};

class TorchTensorSymArg : public TorchSymArg {
 public:
  TorchTensorSymArg(std::string name_);

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
  std::unique_ptr<TorchDtypeSymArg> dtype;
  std::unique_ptr<TorchBoundedIntSymArg> rank;
  std::vector<std::unique_ptr<TorchIntSymArg>> dims;
};

class TorchScalarSymArg : public TorchSymArg {
 public:
  TorchScalarSymArg(std::string name_);

  virtual std::string type() const override;
  virtual std::string var() const override;
  virtual std::string initializer(GEN_MODE mode) const override;

  virtual void resolve_name_conflict(
      std::set<std::string>& names_seen) override;

  virtual std::vector<std::string> gen_arg_setup() const override;
  virtual std::vector<std::string> gen_hard_constraint() const override;
  virtual std::vector<std::string> gen_soft_constraint() const override;
  virtual std::vector<std::string> gen_arg_initialization(
      GEN_MODE mode) const override;

  virtual void assign_concrete_enum_args(std::vector<long>& args) override;
  virtual void assign_concrete_int_args(std::vector<long>& args) override;

 private:
  std::unique_ptr<TorchScalarDtypeSymArg> dtype;
  std::unique_ptr<TorchIntSymArg> intValue;
  std::unique_ptr<TorchUnsignedIntSymArg> uintValue;
  std::unique_ptr<TorchBFloatSymArg> bfloatValue;
  std::unique_ptr<TorchHalfSymArg> halfValue;
  std::unique_ptr<TorchFloatSymArg> floatValue;
  std::unique_ptr<TorchDoubleSymArg> doubleValue;
  std::unique_ptr<TorchHalfSymArg> realValue32;
  std::unique_ptr<TorchHalfSymArg> imaginaryValue32;
  std::unique_ptr<TorchFloatSymArg> realValue64;
  std::unique_ptr<TorchFloatSymArg> imaginaryValue64;
  std::unique_ptr<TorchDoubleSymArg> realValue128;
  std::unique_ptr<TorchDoubleSymArg> imaginaryValue128;
  std::unique_ptr<TorchBoolSymArg> boolValue;
  std::vector<TorchSymArg*> symargs;
};

class TorchOptionalSymArg : public TorchSymArg {
 public:
  TorchOptionalSymArg(std::string name_, std::unique_ptr<TorchSymArg> symarg_);
  TorchOptionalSymArg(std::string name_, std::string base_type_str_);

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

  virtual bool stable() const override;
  std::string base_type() const;

 private:
  std::unique_ptr<TorchBoolSymArg> has_value;
  std::unique_ptr<TorchSymArg> symarg;
  std::string base_type_str;
};

class TorchAPIOptionsSymArg : public TorchSymArg {
 public:
  TorchAPIOptionsSymArg(
      std::string name_, std::string api_optons_class_name_,
      std::vector<std::unique_ptr<TorchSymArg>> ctor_symargs_,
      std::vector<std::unique_ptr<TorchSymArg>> member_symargs_);

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
  std::string api_optons_class_name;
  std::vector<std::unique_ptr<TorchSymArg>> ctor_symargs;
  std::vector<std::unique_ptr<TorchSymArg>> member_symargs;
  std::vector<std::string> member_symarg_setters;

  std::vector<std::string> gen_member_symarg_set(GEN_MODE mode) const;
  std::vector<std::string> gen_api_options_init(GEN_MODE mode) const;
};

#endif
