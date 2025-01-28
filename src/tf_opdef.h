#ifndef PATHFINDER_DRIVER_GENERATOR_TF_OPDEF
#define PATHFINDER_DRIVER_GENERATOR_TF_OPDEF

#include "tf_api.h"

class Input {
 public:
  Input(std::string input_name_, bool is_ref_, std::string size_symbol_,
        std::string datatype_symbol_);
  Input(std::string input_name_, bool is_ref_, std::string size_symbol_,
        DataType allowed_type_);

 private:
  std::string input_name;
  bool is_ref;

  std::string size_symbol;
  std::string datatype_symbol;
  std::optional<DataType> allowed_type;

  friend class OpDef;
};

class Attr {
 public:
  Attr(std::string attr_name_);
  virtual ~Attr() = default;
  const std::string& name() const;
  virtual bool has_default() const = 0;
  virtual std::unique_ptr<TFSymArg> to_symarg() = 0;

 protected:
  std::string attr_name;
};

class AttrType : public Attr {
 public:
  AttrType(std::string attr_name_, std::vector<DataType> allowed_,
           std::optional<DataType> default_val_);
  virtual bool has_default() const;
  virtual std::unique_ptr<TFSymArg> to_symarg();

 private:
  std::vector<DataType> allowed;
  std::optional<DataType> default_val;
  friend class OpDef;
};

class AttrInt : public Attr {
 public:
  AttrInt(std::string attr_name_, std::optional<int> min_,
          std::optional<int> default_val_);
  virtual bool has_default() const;
  virtual std::unique_ptr<TFSymArg> to_symarg();

 private:
  std::optional<int> min;
  std::optional<int> default_val;
  friend class OpDef;
};

class AttrBool : public Attr {
 public:
  AttrBool(std::string attr_name_, std::optional<bool> default_val_);
  virtual bool has_default() const;
  virtual std::unique_ptr<TFSymArg> to_symarg();

 private:
  std::optional<bool> default_val;
  friend class OpDef;
};

class AttrFloat : public Attr {
 public:
  AttrFloat(std::string attr_name_, std::optional<float> default_val_);
  virtual bool has_default() const;
  virtual std::unique_ptr<TFSymArg> to_symarg();

 private:
  std::optional<float> default_val;
  friend class OpDef;
};

class AttrString : public Attr {
 public:
  AttrString(std::string attr_name_, std::vector<std::string> allowed_,
             std::optional<std::string> default_val_);
  virtual bool has_default() const;
  virtual std::unique_ptr<TFSymArg> to_symarg();

 private:
  std::vector<std::string> random_strings();

  std::vector<std::string> allowed;
  std::optional<std::string> default_val;
  friend class OpDef;
};

class AttrShape : public Attr {
 public:
  AttrShape(std::string attr_name_,
            std::optional<std::vector<size_t>> default_val_);
  virtual bool has_default() const;
  virtual std::unique_ptr<TFSymArg> to_symarg();

 private:
  std::optional<std::vector<size_t>> default_val;
  friend class OpDef;
};

class AttrList : public Attr {
 public:
  AttrList(std::string attr_name_, std::vector<std::unique_ptr<Attr>> attrs_,
           size_t size_min_, size_t size_max_,
           std::optional<size_t> default_size_);
  virtual bool has_default() const;
  virtual std::unique_ptr<TFSymArg> to_symarg();

 private:
  std::vector<std::unique_ptr<Attr>> attrs;
  std::optional<size_t> default_size;
  size_t size_min;
  size_t size_max;

  friend class OpDef;
};

class OpDef {
 public:
  OpDef();
  void SetName(std::string op_name_);
  std::string GetName() const;
  void AddInput(std::unique_ptr<Input> input);
  void AddAttr(std::unique_ptr<Attr> attr);
  std::unique_ptr<TFAPI> Resolve();

 private:
  std::string op_name;
  std::vector<std::unique_ptr<Input>> inputs;
  std::vector<std::unique_ptr<Attr>> attrs;
};

std::unique_ptr<Input> parse_input(const std::string& input_signature,
                                   const std::string& op_name);

std::unique_ptr<Attr> parse_attr(
    const std::string& attr_name, const std::string& attr_type,
    const std::vector<std::pair<std::string, std::string>>& properties,
    const std::string& op_name);

std::unique_ptr<Attr> parse_attr(const std::string& attr_signature,
                                 const std::string& op_name);

std::unique_ptr<OpDef> parse_opdef(std::string opdef_line);

#endif
