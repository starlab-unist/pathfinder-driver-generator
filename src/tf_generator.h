#ifndef PATHFINDER_DRIVER_GENERATOR_TF_GENERATOR
#define PATHFINDER_DRIVER_GENERATOR_TF_GENERATOR

#include "generator.h"
#include "tf_api.h"
#include "tf_symarg.h"

class TFFuzzDriverGenerator : public FuzzDriverGenerator<TFAPI> {
 public:
  TFFuzzDriverGenerator();

 private:
  virtual std::vector<std::string> header() const override;
  virtual std::vector<std::string> setup() const override;
  virtual std::vector<std::string> callback() const override;
  virtual std::vector<std::string> footer() const override;
  std::vector<std::string> api_call_code() const;
};

class TFPOVGenerator : public POVGenerator<TFAPI> {
 public:
  TFPOVGenerator();

 private:
  virtual std::vector<std::string> header() const override;
  virtual std::vector<std::string> callback() const override;
  std::vector<std::string> api_call_code() const;
};

#endif
