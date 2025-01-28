#include "tf_generator.h"

#include <cassert>

static const std::string sessionoptions_var = "options";
static const std::string config_var = "config";
static const std::string session_var = "session";
static const std::string graphdef_var = "graph_def";
static const std::string status_var = "status";
static const std::string outputs_var = "outputs";

TFFuzzDriverGenerator::TFFuzzDriverGenerator() : FuzzDriverGenerator<TFAPI>() {}
std::vector<std::string> TFFuzzDriverGenerator::header() const {
  return {
      "#include \"tensorflow/cc/framework/scope.h\"",
      "#include \"tensorflow/core/graph/graph.h\"",
      "#include \"tensorflow/core/public/session.h\"",
      "#include \"tensorflow/cc/ops/array_ops.h\"",
      "#include \"tensorflow/cc/ops/standard_ops.h\"",
      "#include \"tensorflow/core/kernels/pathfinder/fuzzer_util.h\"",
      "#include \"pathfinder.h\"\n",

      "using namespace tensorflow;",
      "using namespace fuzzer_util;\n",
  };
}
std::vector<std::string> TFFuzzDriverGenerator::setup() const {
  std::vector<std::string> setup_code;

  setup_code.push_back(api->pathfinder_setup_sig() + " {");

  concat(setup_code, "  ", arg_setup_code());
  concat(setup_code, "  ", hard_constraint_code());
  concat(setup_code, "  ", soft_constraint_code());

  setup_code.push_back("}\n");

  return setup_code;
}
std::vector<std::string> TFFuzzDriverGenerator::callback() const {
  std::vector<std::string> callback_code;

  callback_code.push_back(api->pathfinder_test_one_input_sig() + " {");
  std::vector<std::string> basic_setup = {
      "SessionOptions " + sessionoptions_var + ";",
      "ConfigProto & " + config_var + " = " + sessionoptions_var + ".config;",
      config_var + ".set_inter_op_parallelism_threads(1);",
      config_var + ".set_intra_op_parallelism_threads(1);",
      config_var + ".set_use_per_session_threads(false);",
      "std::unique_ptr<tensorflow::Session>",
      "  " + session_var + "(tensorflow::NewSession(" + sessionoptions_var +
          "));\n",
  };
  concat(callback_code, "  ", basic_setup);

  concat(callback_code, "  ", input_pass_condition_code());

  concat(callback_code, "  ", arg_initialization_code());

  callback_code.push_back("  PathFinderExecuteTarget(");
  concat(callback_code, "    ", api_call_code());

  callback_code.push_back("    Status " + status_var + ";");
  callback_code.push_back("    GraphDef " + graphdef_var + ";");
  callback_code.push_back("    " + status_var + " = " + tf_scope_var +
                          ".ToGraphDef(&" + graphdef_var + ");");
  callback_code.push_back("    if (" + status_var + ".ok()) {");
  callback_code.push_back("      " + status_var + " = " + session_var +
                          "->Create(" + graphdef_var + ");");
  callback_code.push_back("      if (" + status_var + ".ok()) {");
  callback_code.push_back("        std::vector<Tensor> " + outputs_var + ";");
  callback_code.push_back("        " + status_var + " = " + session_var +
                          "->Run({}, {\"" + target_var + "\"}, {}, &" +
                          outputs_var + ");");
  callback_code.push_back("      }");
  callback_code.push_back("    }");
  callback_code.push_back("  );\n");

  callback_code.push_back("  if (!" + status_var + ".ok())");
  callback_code.push_back("    return -2;\n");

  callback_code.push_back("  return 0;");
  callback_code.push_back("}\n");

  return callback_code;
}
std::vector<std::string> TFFuzzDriverGenerator::api_call_code() const {
  std::string module_init =
      "auto " + target_var + assign + api->get_qualified_name() + "(";
  module_init += api->get_scope()->expr(MODE_DRIVER) + ".WithOpName(\"" +
                 target_var + "\")";
  for (size_t i = 0; i < api->get_ordinary_symargs().size(); i++)
    module_init += comma + api->get_ordinary_symargs()[i]->expr(MODE_DRIVER);
  module_init += ");\n";

  return {module_init};
}
std::vector<std::string> TFFuzzDriverGenerator::footer() const { return {}; }

TFPOVGenerator::TFPOVGenerator() : POVGenerator<TFAPI>() {}
std::vector<std::string> TFPOVGenerator::header() const {
  return {
      "#include \"tensorflow/cc/framework/scope.h\"",
      "#include \"tensorflow/core/graph/graph.h\"",
      "#include \"tensorflow/core/public/session.h\"",
      "#include \"tensorflow/cc/ops/array_ops.h\"",
      "#include \"tensorflow/cc/ops/standard_ops.h\"",
      "#include \"tensorflow/core/kernels/pathfinder/fuzzer_util.h\"",

      "using namespace tensorflow;",
      "using namespace fuzzer_util;\n",
  };
}
std::vector<std::string> TFPOVGenerator::callback() const {
  std::vector<std::string> callback_code;

  callback_code.push_back("int main() {");
  std::vector<std::string> basic_setup = {
      "SessionOptions " + sessionoptions_var + ";",
      "ConfigProto & " + config_var + " = " + sessionoptions_var + ".config;",
      config_var + ".set_inter_op_parallelism_threads(1);",
      config_var + ".set_intra_op_parallelism_threads(1);",
      config_var + ".set_use_per_session_threads(false);",
      "std::unique_ptr<tensorflow::Session>",
      "  " + session_var + "(tensorflow::NewSession(" + sessionoptions_var +
          "));\n",
  };
  concat(callback_code, "  ", basic_setup);

  concat(callback_code, "  ", arg_initialization_code());

  concat(callback_code, "  ", api_call_code());

  callback_code.push_back("  Status " + status_var + ";");
  callback_code.push_back("  GraphDef " + graphdef_var + ";");
  callback_code.push_back("  " + status_var + " = " + tf_scope_var +
                          ".ToGraphDef(&" + graphdef_var + ");");
  callback_code.push_back("  if (" + status_var + ".ok()) {");
  callback_code.push_back("    " + status_var + " = " + session_var +
                          "->Create(" + graphdef_var + ");");
  callback_code.push_back("    if (" + status_var + ".ok()) {");
  callback_code.push_back("      std::vector<Tensor> " + outputs_var + ";");
  callback_code.push_back("      " + status_var + " = " + session_var +
                          "->Run({}, {\"" + target_var + "\"}, {}, &" +
                          outputs_var + ");");
  callback_code.push_back("    }");
  callback_code.push_back("  }\n");

  callback_code.push_back("  if (!" + status_var + ".ok())");
  callback_code.push_back("    LOG(WARNING) << " + status_var +
                          ".message();\n");

  callback_code.push_back("  return 0;");
  callback_code.push_back("}\n");

  return callback_code;
}
std::vector<std::string> TFPOVGenerator::api_call_code() const {
  std::string module_init =
      "auto " + target_var + assign + api->get_qualified_name() + "(";
  module_init +=
      api->get_scope()->expr(MODE_POV) + ".WithOpName(\"" + target_var + "\")";
  for (size_t i = 0; i < api->get_ordinary_symargs().size(); i++)
    module_init += comma + api->get_ordinary_symargs()[i]->expr(MODE_POV);
  module_init += ");\n";

  return {module_init};
}
