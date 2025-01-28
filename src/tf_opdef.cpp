#include "tf_opdef.h"

#include <cassert>
#include <climits>

#include "tf_cpp_apis.h"

Input::Input(std::string input_name_, bool is_ref_, std::string size_symbol_,
             std::string datatype_symbol_)
    : input_name(input_name_),
      is_ref(is_ref_),
      size_symbol(size_symbol_),
      datatype_symbol(datatype_symbol_) {}
Input::Input(std::string input_name_, bool is_ref_, std::string size_symbol_,
             DataType allowed_type_)
    : input_name(input_name_),
      is_ref(is_ref_),
      size_symbol(size_symbol_),
      allowed_type(allowed_type_) {}

Attr::Attr(std::string attr_name_) : attr_name(attr_name_) {}
const std::string& Attr::name() const { return attr_name; }

AttrType::AttrType(std::string attr_name_, std::vector<DataType> allowed_,
                   std::optional<DataType> default_val_)
    : Attr(attr_name_) {
  if (allowed_.empty()) {
    allowed = all_datatypes();
  } else {
    allowed = allowed_;
  }
  default_val = default_val_;
}
bool AttrType::has_default() const { return default_val.has_value(); }
std::unique_ptr<TFSymArg> AttrType::to_symarg() {
  assert(!allowed.empty());
  std::vector<std::string> allowed_str;
  for (auto& dt : allowed) {
    allowed_str.push_back(string_from_datatype(dt));
  }
  return std::make_unique<TFDtypeSymArg>(attr_name, allowed_str);
}

AttrInt::AttrInt(std::string attr_name_, std::optional<int> min_,
                 std::optional<int> default_val_)
    : Attr(attr_name_) {
  min = min_;
  default_val = default_val_;
}
bool AttrInt::has_default() const { return default_val.has_value(); }
std::unique_ptr<TFSymArg> AttrInt::to_symarg() {
  return std::make_unique<TFIntSymArg>(attr_name,
                                       min.has_value() ? min.value() : 1);
}

AttrBool::AttrBool(std::string attr_name_, std::optional<bool> default_val_)
    : Attr(attr_name_) {
  default_val = default_val_;
}
bool AttrBool::has_default() const { return default_val.has_value(); }
std::unique_ptr<TFSymArg> AttrBool::to_symarg() {
  return std::make_unique<TFBoolSymArg>(attr_name);
}

AttrFloat::AttrFloat(std::string attr_name_, std::optional<float> default_val_)
    : Attr(attr_name_) {
  default_val = default_val_;
}
bool AttrFloat::has_default() const { return default_val.has_value(); }
std::unique_ptr<TFSymArg> AttrFloat::to_symarg() {
  return std::make_unique<TFFloatSymArg>(attr_name);
}

AttrString::AttrString(std::string attr_name_,
                       std::vector<std::string> allowed_,
                       std::optional<std::string> default_val_)
    : Attr(attr_name_) {
  allowed = allowed_;
  default_val = default_val_;
}
bool AttrString::has_default() const { return default_val.has_value(); }
std::unique_ptr<TFSymArg> AttrString::to_symarg() {
  if (allowed.empty())
    return std::make_unique<TFStringSymArg>(attr_name, random_strings());
  return std::make_unique<TFStringSymArg>(attr_name, allowed);
}
std::vector<std::string> AttrString::random_strings() {
  return {
      "", "1", "_", "a", std::string(256, 'a'),
  };
}

AttrShape::AttrShape(std::string attr_name_,
                     std::optional<std::vector<size_t>> default_val_)
    : Attr(attr_name_) {
  default_val = default_val_;
}
bool AttrShape::has_default() const { return default_val.has_value(); }
std::unique_ptr<TFSymArg> AttrShape::to_symarg() {
  return std::make_unique<TFPartialTensorShapeSymArg>(attr_name);
}

AttrList::AttrList(std::string attr_name_,
                   std::vector<std::unique_ptr<Attr>> attrs_, size_t size_min_,
                   size_t size_max_, std::optional<size_t> default_size_)
    : Attr(attr_name_) {
  attrs = std::move(attrs_);
  default_size = default_size_;
  size_min = size_min_;
  size_max = size_max_;
}
bool AttrList::has_default() const { return default_size.has_value(); }
std::unique_ptr<TFSymArg> AttrList::to_symarg() {
  std::vector<std::unique_ptr<TFSymArg>> symargs;
  for (auto& attr : attrs) {
    auto symarg = attr->to_symarg();
    if (auto* string_symarg = dynamic_cast<TFStringSymArg*>(symarg.get()))
      string_symarg->to_tstring();
    symargs.push_back(std::move(symarg));
  }
  return std::make_unique<TFArraySliceSymArg>(attr_name, std::move(symargs),
                                              size_min, size_max);
}

OpDef::OpDef() {}
void OpDef::SetName(std::string op_name_) { op_name = op_name_; }
std::string OpDef::GetName() const { return op_name; }
void OpDef::AddInput(std::unique_ptr<Input> input) {
  inputs.push_back(std::move(input));
}
void OpDef::AddAttr(std::unique_ptr<Attr> attr) {
  attrs.push_back(std::move(attr));
}
std::unique_ptr<TFAPI> OpDef::Resolve() {
  std::vector<std::unique_ptr<TFSymArg>> dependent;
  std::vector<std::unique_ptr<TFSymArg>> required;
  std::vector<std::unique_ptr<TFSymArg>> optional;

  std::map<std::string, TFBoundedIntSymArg*> symbolic_numbers;
  std::map<std::string, TFDtypeSymArg*> symbolic_datatypes;

  for (auto& input : inputs) {
    if (input->size_symbol != "")
      symbolic_numbers[input->size_symbol] = nullptr;
    if (input->datatype_symbol != "") {
      symbolic_datatypes[input->datatype_symbol] = nullptr;
    }
  }

  for (auto& attr : attrs) {
    if (symbolic_numbers.find(attr->name()) != symbolic_numbers.end()) {
      AttrInt* int_attr = dynamic_cast<AttrInt*>(attr.get());
      assert(int_attr != nullptr);
      std::unique_ptr<TFBoundedIntSymArg> symbolic_number;
      if (int_attr->default_val.has_value() &&
          int_attr->default_val.value() != 0) {
        symbolic_number = std::make_unique<TFBoundedIntSymArg>(
            int_attr->name(), int_attr->default_val.value(), 1);
      } else if (int_attr->min.has_value()) {
        int min = int_attr->min.value();
        assert(0 <= min &&
               min <=
                   TF_MAX_ARRAY_SIZE);  // Size of InputList is always positive
        size_t size = TF_MAX_ARRAY_SIZE - min + 1;
        symbolic_number = std::make_unique<TFBoundedIntSymArg>(
            int_attr->name(), int_attr->min.value(), size);
      } else {
        symbolic_number = std::make_unique<TFBoundedIntSymArg>(
            int_attr->name(), 0, TF_MAX_ARRAY_SIZE + 1);
      }
      symbolic_numbers[attr->name()] = symbolic_number.get();
      dependent.push_back(std::move(symbolic_number));
    } else if (symbolic_datatypes.find(attr->name()) !=
               symbolic_datatypes.end()) {
      AttrType* dtype_attr = dynamic_cast<AttrType*>(attr.get());
      assert(dtype_attr != nullptr);
      std::vector<std::string> allowed_str;
      for (auto& dt : dtype_attr->allowed) {
        if (datatype_not_supported(dt)) continue;
        allowed_str.push_back(string_from_datatype(dt));
      }
      if (allowed_str.empty()) {
        // std::cerr << "UNSUPPORTED TENSOR TYPE" << dtype_attr->name()
        //           << std::endl;
        return nullptr;
      }
      std::unique_ptr<TFDtypeSymArg> symbolic_datatype =
          std::make_unique<TFDtypeSymArg>(dtype_attr->name(), allowed_str);
      symbolic_datatypes[attr->name()] = symbolic_datatype.get();
      dependent.push_back(std::move(symbolic_datatype));
    }
  }

  for (auto& input : inputs) {
    TFBoundedIntSymArg* symbolic_number =
        input->size_symbol == "" ? nullptr
                                 : symbolic_numbers[input->size_symbol];
    TFDtypeSymArg* symbolic_datatype =
        input->datatype_symbol == ""
            ? nullptr
            : symbolic_datatypes[input->datatype_symbol];
    if (symbolic_number != nullptr) {
      if (symbolic_datatype != nullptr) {
        required.push_back(std::make_unique<TFInputListSymArg>(
            input->input_name, input->is_ref, symbolic_datatype,
            symbolic_number));
      } else {
        assert(input->allowed_type.has_value());
        if (datatype_not_supported(input->allowed_type.value())) {
          // std::cerr << "UNSUPPORTED TYPE OF TENSOR " << input->input_name
          //           << std::endl;
          return nullptr;
        }
        std::vector<std::string> allowed = {
            string_from_datatype(input->allowed_type.value())};
        auto dtype = std::make_unique<TFDtypeSymArg>(
            input->input_name + "_dtype", allowed);
        required.push_back(std::make_unique<TFInputListSymArg>(
            input->input_name, input->is_ref, std::move(dtype),
            symbolic_number));
      }
    } else if (symbolic_datatype != nullptr) {
      required.push_back(std::make_unique<TFTensorSymArg>(
          input->input_name, input->is_ref, symbolic_datatype));
    } else {
      assert(input->allowed_type.has_value());
      if (datatype_not_supported(input->allowed_type.value())) {
        // std::cerr << "UNSUPPORTED TYPE OF TENSOR " << input->input_name
        //           << std::endl;
        return nullptr;
      }
      std::vector<std::string> allowed = {
          string_from_datatype(input->allowed_type.value())};
      auto dtype = std::make_unique<TFDtypeSymArg>(input->input_name + "_dtype",
                                                   allowed);
      required.push_back(std::make_unique<TFTensorSymArg>(
          input->input_name, input->is_ref, std::move(dtype)));
    }
  }

  for (auto& attr : attrs) {
    if (symbolic_numbers.find(attr->name()) != symbolic_numbers.end()) {
      continue;
    } else if (symbolic_datatypes.find(attr->name()) !=
               symbolic_datatypes.end()) {
      continue;
    } else if (!attr->has_default()) {
      required.push_back(attr->to_symarg());
    } else {
      optional.push_back(attr->to_symarg());
    }
  }
  return std::make_unique<TFAPI>(op_name, std::move(dependent),
                                 std::move(required), std::move(optional));
}

std::unique_ptr<Input> parse_input(const std::string& input_signature,
                                   const std::string& op_name) {
  std::vector<std::string> name_type_pair = split(input_signature, ":");
  assert(name_type_pair.size() == 2);
  std::string input_name = strip(name_type_pair[0]);
  std::string datatype_str = strip(name_type_pair[1]);
  bool is_ref = false;
  if (startswith(datatype_str, "Ref(")) {
    datatype_str = strip_prefix(datatype_str, "Ref(");
    datatype_str = strip_suffix(datatype_str, ")");
    is_ref = true;
  }

  if (datatype_str == "") {
    return nullptr;
  }

  std::string size_symbol;
  std::vector<std::string> size_type_pair = split(datatype_str, "*");

  if (size_type_pair.size() > 1) {
    assert(size_type_pair.size() == 2);
    size_symbol = strip(size_type_pair[0]);
    datatype_str = strip(size_type_pair[1]);
  }

  if (auto allowed_type = datatype_from_string(datatype_str)) {
    return std::make_unique<Input>(input_name, is_ref, size_symbol,
                                   allowed_type.value());
  } else {
    return std::make_unique<Input>(input_name, is_ref, size_symbol,
                                   datatype_str);
  }
  UNREACHABLE;
}

std::unique_ptr<Attr> parse_attr(
    const std::string& attr_name, const std::string& attr_type,
    const std::vector<std::pair<std::string, std::string>>& properties,
    const std::string& op_name) {
  static const std::set<std::string> ignore_list = {
      "func",
      "tensor",
  };

  std::unique_ptr<Attr> retval;

  if (attr_type == "type") {
    std::optional<DataType> default_val;
    std::vector<DataType> allowed;
    for (auto& p : properties) {
      std::string key = p.first;
      std::string value = p.second;
      if (key == "default") {
        default_val = datatype_from_string(value);
        assert(default_val.has_value());
      } else if (key == "allowed") {
        assert(value != "");
        for (auto& type_str : split(value, ",")) {
          std::optional<DataType> datatype =
              datatype_from_string(strip(type_str));
          assert(datatype.has_value());
          allowed.push_back(datatype.value());
        }
      } else {
        // std::cout << "UNUSED PROPERTY `" << key << "` of attr type `"
        //           << attr_type << "` (Op: " << op_name << ")" << std::endl;
        continue;
      }
    }
    retval = std::make_unique<AttrType>(attr_name, allowed, default_val);
  } else if (attr_type == "int") {
    std::optional<int> default_val;
    std::optional<int> min;
    for (auto& p : properties) {
      std::string key = p.first;
      std::string value = p.second;
      if (key == "default") {
        try {
          default_val = std::stoi(value);
        } catch (const std::out_of_range& e) {
          default_val = INT_MAX;
        }
      } else if (key == "min") {
        min = std::stoi(value);
      } else {
        // std::cout << "UNUSED PROPERTY `" << key << "` of attr type `"
        //           << attr_type << "` (Op: " << op_name << ")" << std::endl;
        continue;
      }
    }
    retval = std::make_unique<AttrInt>(attr_name, min, default_val);
  } else if (attr_type == "bool") {
    std::optional<bool> default_val;
    for (auto& p : properties) {
      std::string key = p.first;
      std::string value = p.second;
      if (key == "default") {
        if (value == "true") {
          default_val = true;
        } else if (value == "false") {
          default_val = false;
        } else {
          UNREACHABLE;
        }
      } else {
        // std::cout << "UNUSED PROPERTY `" << key << "` of attr type `"
        //           << attr_type << "` (Op: " << op_name << ")" << std::endl;
        continue;
      }
    }
    retval = std::make_unique<AttrBool>(attr_name, default_val);
  } else if (attr_type == "float") {
    std::optional<float> default_val;
    for (auto& p : properties) {
      std::string key = p.first;
      std::string value = p.second;
      if (key == "default") {
        default_val = std::stof(value);
      } else {
        // std::cout << "UNUSED PROPERTY `" << key << "` of attr type `"
        //           << attr_type << "` (Op: " << op_name << ")" << std::endl;
        continue;
      }
    }
    retval = std::make_unique<AttrFloat>(attr_name, default_val);
  } else if (attr_type == "string") {
    std::optional<std::string> default_val;
    std::vector<std::string> allowed;
    for (auto& p : properties) {
      std::string key = p.first;
      std::string value = p.second;
      if (key == "default") {
        assert(startswith(value, "\""));
        assert(endswith(value, "\""));
        std::string str = value;
        str = strip_prefix(str, "\"");
        str = strip_suffix(str, "\"");
        default_val = str;
      } else if (key == "allowed") {
        assert(value != "");
        for (auto& val : split(value, ",")) {
          std::string str = strip(val);
          assert(!strip(str).empty());
          assert(startswith(str, "\""));
          assert(endswith(str, "\""));
          str = strip_prefix(str, "\"");
          str = strip_suffix(str, "\"");
          allowed.push_back(strip(str));
        }
      } else {
        // std::cout << "UNUSED PROPERTY `" << key << "` of attr type `"
        //           << attr_type << "` (Op: " << op_name << ")" << std::endl;
        continue;
      }
    }
    retval = std::make_unique<AttrString>(attr_name, allowed, default_val);
  } else if (attr_type == "shape") {
    std::optional<std::vector<size_t>> default_val;
    for (auto& p : properties) {
      std::string key = p.first;
      std::string value = p.second;
      if (key == "default") {
        std::vector<size_t> default_shape = {};
        if (value != "" && value != "<unknown>") {
          for (auto& dim_str : split(value, ",")) {
            default_shape.push_back(std::stoi(dim_str));
          }
        }
        default_val = default_shape;
      } else {
        // std::cout << "UNUSED PROPERTY `" << key << "` of attr type `"
        //           << attr_type << "` (Op: " << op_name << ")" << std::endl;
        continue;
      }
    }
    retval = std::make_unique<AttrShape>(attr_name, default_val);
  } else if (startswith(attr_type, "list")) {
    std::string base_type = attr_type;
    base_type = strip_prefix(base_type, "list(");
    base_type = strip_suffix(base_type, ")");
    std::optional<size_t> default_size;
    size_t size_min = 0;
    size_t size_max = TF_MAX_ARRAY_SIZE;
    std::vector<std::pair<std::string, std::string>> properties_sub;
    for (auto& p : properties) {
      std::string key = p.first;
      std::string value = p.second;
      if (key == "default") {
        default_size = strip(value) == "" ? 0 : split(value, ",").size();
        if (default_size.value() > 0) {
          size_min = default_size.value();
          size_max = default_size.value();
        }
      } else if (key == "min") {
        size_min = std::stoi(value);
        assert(size_min <= TF_MAX_ARRAY_SIZE);
        if (size_min > 1) size_max = size_min;
      } else {
        properties_sub.push_back({key, value});
      }
    }
    std::vector<std::unique_ptr<Attr>> attrs;
    for (size_t i = 0; i < size_max; i++)
      attrs.push_back(parse_attr(attr_name + "_" + std::to_string(i), base_type,
                                 properties_sub, op_name));
    retval = std::make_unique<AttrList>(attr_name, std::move(attrs), size_min,
                                        size_max, default_size);
  } else if (ignore_list.find(attr_type) != ignore_list.end()) {
    // Ignore
  } else {
    std::cerr << "UNHANDLED ATTR TYPE: " << attr_type << std::endl;
    PDG_ASSERT(false);
  }
  return retval;
}

std::unique_ptr<Attr> parse_attr(const std::string& attr_signature,
                                 const std::string& op_name) {
  std::unique_ptr<Attr> retval;

  std::string name_type_str, properties_str;
  std::tie(name_type_str, properties_str) = lsplit(attr_signature, ",");
  std::vector<std::string> name_type_pair = split(name_type_str, ":");
  assert(name_type_pair.size() == 2);
  std::string attr_name = strip(name_type_pair[0]);
  std::string attr_type = strip(name_type_pair[1]);

  std::vector<std::pair<std::string, std::string>> properties;
  while (!strip(properties_str).empty()) {
    std::string property_key, property_value;
    std::tie(property_key, properties_str) = lsplit(properties_str, "=");

    if (startswith(properties_str, "[")) {
      std::tie(property_value, properties_str) =
          lsplit(properties_str.substr(1), "]");
      properties_str = lsplit(properties_str, ",").second;
    } else if (startswith(properties_str, "\"")) {
      std::tie(property_value, properties_str) =
          lsplit(properties_str.substr(1), "\"");
      property_value = "\"" + property_value + "\"";
      properties_str = lsplit(properties_str, ",").second;
    } else {
      std::tie(property_value, properties_str) = lsplit(properties_str, ",");
    }
    properties.push_back({strip(property_key), strip(property_value)});
  }

  for (auto property_str : split(properties_str, ",")) {
    property_str = strip(property_str);
    if (property_str.empty()) break;
    std::vector<std::string> property = split(property_str, "=");
    if (property.size() != 2) {
      // std::cout << attr_signature << std::endl;
      // std::cout << properties_str << std::endl;
      PDG_ASSERT(false);
    }
    properties.push_back({strip(property[0]), strip(property[1])});
  }

  return parse_attr(attr_name, attr_type, properties, op_name);
}

std::unique_ptr<OpDef> parse_opdef(std::string opdef_line) {
  static const std::set<std::string> ignore_list = {
      "is_commutative",
      "is_aggregate",
      "is_stateful",
      "allows_uninitialized_input",
      "is_distributed_communication",
  };

  std::unique_ptr<OpDef> opdef = std::make_unique<OpDef>();

  opdef_line = strip_prefix(opdef_line, "Op<");
  opdef_line = strip_suffix(opdef_line, ">");

  std::string name_field;
  std::tie(name_field, opdef_line) = lsplit(opdef_line, ";");
  std::string name_key, name_value;
  std::tie(name_key, name_value) = lsplit(name_field, "=");
  assert(name_key == "name");
  std::string op_name = name_value;

  if (tf_cpp_apis::is_cpp_api(op_name)) {
    opdef->SetName(op_name);
  } else {
    return nullptr;
  }
  // std::cout << "Parsing API `" << op_name << "`..." << std::endl;

  for (auto field : split(opdef_line, ";")) {
    auto key_value_pair = lsplit(field, "=");
    if (key_value_pair.second == "") {
      // std::cout << "=================== Error ===================" <<
      // std::endl; std::cout << opdef_line << std::endl;
      PDG_ASSERT(false);
    }
    std::string key = strip(key_value_pair.first);
    std::string value = strip(key_value_pair.second);

    if (key == "signature") {
      std::vector<std::string> inputs_outputs_pair = split(value, "->");
      assert(inputs_outputs_pair.size() == 2);
      std::string inputs = strip(inputs_outputs_pair[0]);
      if (!inputs.empty())
        for (auto input_signature : split(inputs, ",")) {
          std::unique_ptr<Input> input = parse_input(input_signature, op_name);
          if (input == nullptr) {
            // std::cerr << "Parsing Failed: " << op_name << std::endl;
            return nullptr;
          }
          opdef->AddInput(std::move(input));
        }
    } else if (key == "attr") {
      std::unique_ptr<Attr> attr = parse_attr(value, op_name);
      if (attr == nullptr) {
        // std::cerr << "Ignore Op `" << op_name << "`" << std::endl;
      } else {
        opdef->AddAttr(std::move(attr));
      }
    } else if (ignore_list.find(key) != ignore_list.end()) {
      continue;
    } else {
      std::cerr << "Unknown Key `" << key << "`." << std::endl;
      PDG_ASSERT(false);
    }
  }
  return opdef;
}
