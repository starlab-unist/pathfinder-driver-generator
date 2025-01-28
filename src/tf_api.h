#ifndef PATHFINDER_DRIVER_GENERATOR_TF_API
#define PATHFINDER_DRIVER_GENERATOR_TF_API

#include "api.h"
#include "tf_symarg.h"

extern const std::string target_var;

class TFAPI : public API {
 public:
  TFAPI(std::string api_name_,
        std::vector<std::unique_ptr<TFSymArg>> dependent_symargs_,
        std::vector<std::unique_ptr<TFSymArg>> required_symargs_,
        std::vector<std::unique_ptr<TFSymArg>> optional_symargs_);
  std::string get_qualified_name() const;
  std::string get_unqualified_name() const;
  std::string pathfinder_setup_sig() const;
  std::string pathfinder_test_one_input_sig() const;
  TFScopeSymArg* get_scope() const;
  const std::vector<TFSymArg*>& get_ordinary_symargs() const;

 private:
  TFScopeSymArg* scope;
  std::vector<TFSymArg*> dependent_symargs;
  std::vector<TFSymArg*> ordinary_symargs;
};

#endif
