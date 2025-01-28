#include <cassert>
#include <climits>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>

#include "options.h"
#include "tf_generator.h"
#include "tf_opdef.h"
#include "utils.h"
namespace fs = std::filesystem;

class TFOutput {
 public:
  TFOutput() {}
  void add(std::unique_ptr<TFAPI> api) { apis.push_back(std::move(api)); }
  void write_driver(const fs::path& target_dir_path) {
    std::set<std::string> names_seen;
    create_directories(target_dir_path);

    TFFuzzDriverGenerator generator;

    for (auto& api : apis) {
      fs::path driver_file_path =
          target_dir_path / (api->get_unqualified_name() + ".cc");
      write_file(driver_file_path.string(), generator.gen(api.get()));
    }

    fs::path main_file_path = target_dir_path / "pathfinder_driver_main.cc";
    write_file(main_file_path.string(), main_contents());

    fs::path bazel_file_path = target_dir_path / "BUILD";
    write_file(bazel_file_path.string(), bazel_contents());
  }
  void write_pov(const fs::path& target_dir_path,
                 std::vector<long> concrete_args) {
    std::set<std::string> names_seen;
    create_directories(target_dir_path);

    TFPOVGenerator generator;

    for (auto& api : apis) {
      fs::path driver_file_path =
          target_dir_path / (api->get_unqualified_name() + ".cc");
      api->assign(concrete_args);
      write_file(driver_file_path.string(), generator.gen(api.get()));
    }

    fs::path main_file_path = target_dir_path / "pathfinder_driver_main.cc";
    write_file(main_file_path.string(), main_contents());

    fs::path bazel_file_path = target_dir_path / "BUILD";
    write_file(bazel_file_path.string(), bazel_contents());
  }

 private:
  std::vector<std::unique_ptr<TFAPI>> apis;
  std::string main_contents() const {
    std::string pf_wrapper_decls;
    for (auto& api : apis) {
      pf_wrapper_decls += api->pathfinder_setup_sig() + ";\n";
      pf_wrapper_decls += api->pathfinder_test_one_input_sig() + ";\n";
    }

    std::string select_setup = "  ";
    std::string select_callback = "  ";
    for (auto& api : apis) {
      std::string api_name = api->get_unqualified_name();
      select_setup += "if (target_api == \"" + api_name +
                      "\") {\n"
                      "    PathFinderSetup_Wrapper = PathFinderSetup_" +
                      api_name +
                      ";\n"
                      "  } else ";
      select_callback +=
          "if (target_api == \"" + api_name +
          "\") {\n"
          "    PathFinderTestOneInput_Wrapper = PathFinderTestOneInput_" +
          api_name +
          ";\n"
          "  } else ";
    }

    std::string contents =
        "#include \"pathfinder.h\"\n\n" + pf_wrapper_decls + "\n" +

        "void (*PathFinderSetup_Wrapper)();\n"
        "int (*PathFinderTestOneInput_Wrapper)(const pathfinder::Input& x);\n\n"

        "void select_setup(const std::string& target_api) {\n" +
        select_setup +
        "{\n"
        "    std::cerr << \"Target API `\" << target_api << \"` is not valid\" "
        "<< std::endl;\n"
        "    exit(0);\n"
        "  }\n"
        "}\n"
        "void select_callback(const std::string& target_api) {\n" +
        select_callback +
        "{\n"
        "    std::cerr << \"Target API `\" << target_api << \"` is not valid\" "
        "<< std::endl;\n"
        "    exit(0);\n"
        "  }\n"
        "}\n\n"

        "extern \"C\" {\n\n"

        "void PathFinderSetup() {\n" +
        "  PathFinderSetup_Wrapper();\n"
        "}\n"
        "int PathFinderTestOneInput(const pathfinder::Input& x) {\n" +
        "  return PathFinderTestOneInput_Wrapper(x);\n"
        "}\n\n"

        "}  // extern \"C\"\n\n" +

        "int main(int argc, char **argv) {\n"
        "  std::string target_api = argv[1];\n"
        "  select_setup(target_api);\n"
        "  select_callback(target_api);\n\n"

        "  pathfinder::parse_arg(argc, argv);\n"
        "  return pathfinder::driver(PathFinderTestOneInput);\n"
        "}\n";

    return contents;
  }
  std::string bazel_contents() const {
    std::string contents =
        "load(\"//tensorflow/core/platform:rules_cc.bzl\", \"cc_library\")\n"
        "load(\"//tensorflow/core/kernels/pathfinder:pathfinder.bzl\", "
        "\"pathfinder_fuzz_driver\")\n"
        "load(\"//tensorflow:tensorflow.bzl\", \"tf_cc_binary\")\n\n"

        "package(\n"
        "    default_visibility = "
        "[\"//tensorflow/core/kernels/pathfinder/driver:__pkg__\"],\n"
        ")\n\n";

    for (auto& api : apis) {
      contents +=
          "pathfinder_fuzz_driver(\"" + api->get_unqualified_name() + "\")\n";
    }

    contents +=
        "tf_cc_binary(\n"
        "    name = \"pathfinder_driver_main\",\n"
        "    srcs = [\n"
        "        \"pathfinder_driver_main.cc\",\n"
        "    ],\n"
        "    deps = [\n";

    for (auto& api : apis) {
      contents += "        \":" + api->get_unqualified_name() + "\",\n";
    }

    contents +=
        "    ],\n"
        ")\n";

    return contents;
  }
};

void gen_tf_drivers(const fs::path& output_dir) {
  fs::path output_dir_ = output_dir;
  if (output_dir_.empty()) output_dir_ = "./pathfinder-tf";

  std::string file_name = "tf" + DLL_VERSION + ".txt";
  fs::path tf_api_sig_filepath = api_signatures_dir_path() / file_name;
  std::ifstream tf_api_sig_file(tf_api_sig_filepath);

  TFOutput output;
  std::string buffer;
  while (getline(tf_api_sig_file, buffer)) {
    std::string line = strip(buffer);
    if (line == "" || startswith(line, "#")) continue;
    if (auto opdef = parse_opdef(line)) {
      auto api = opdef->Resolve();
      if (api == nullptr) {
        // std::cout << "Driver Generation Failed: " << opdef->GetName()
        //           << std::endl;
        continue;
      }
      output.add(std::move(api));
    }
  }

  std::string template_dir_name = "tf" + DLL_VERSION;
  fs::path tf_template_dir_path = template_dir_path() / template_dir_name;
  cp_dir_recursive(tf_template_dir_path, output_dir_);
  fs::path drivers_dir(output_dir_ / "driver");
  output.write_driver(drivers_dir);
}

void gen_tf_pov(const std::string& target_api, std::vector<long> concrete_args,
                const fs::path& output_dir) {
  fs::path output_dir_ = output_dir;
  if (output_dir_.empty()) output_dir_ = "./pathfinder-tf";

  std::string file_name = "tf" + DLL_VERSION + ".txt";
  fs::path tf_api_sig_filepath = api_signatures_dir_path() / file_name;
  std::ifstream tf_api_sig_file(tf_api_sig_filepath);

  TFOutput output;
  std::string buffer;
  while (getline(tf_api_sig_file, buffer)) {
    std::string line = strip(buffer);
    if (line == "" || startswith(line, "#")) continue;
    if (auto opdef = parse_opdef(line)) {
      auto api = opdef->Resolve();
      if (api == nullptr) {
        // std::cout << "Driver Generation Failed: " << opdef->GetName()
        //           << std::endl;
        continue;
      }
      if (api->get_name() == target_api) {
        output.add(std::move(api));
        break;
      }
    }
  }

  std::string template_dir_name = "tf" + DLL_VERSION;
  fs::path tf_template_dir_path = template_dir_path() / template_dir_name;
  cp_dir_recursive(tf_template_dir_path, output_dir_);
  fs::path drivers_dir(output_dir_ / "pov");
  output.write_pov(drivers_dir, concrete_args);
}

std::unique_ptr<TFAPI> get_api(const std::string& api_name) {
  std::string file_name = "tf" + DLL_VERSION + ".txt";
  fs::path tf_api_sig_filepath = api_signatures_dir_path() / file_name;
  std::ifstream tf_api_sig_file(tf_api_sig_filepath);

  std::string buffer;
  while (getline(tf_api_sig_file, buffer)) {
    std::string line = strip(buffer);
    if (line == "" || startswith(line, "#")) continue;
    if (auto opdef = parse_opdef(line)) {
      auto api = opdef->Resolve();
      if (api == nullptr) continue;
      if (api->get_name() == api_name) {
        return std::move(api);
      }
    }
  }
  PDG_CHECK(false, "Failed to find API `" + api_name + "`.");
}

void gen_tf_pov(const fs::path& buggy_input_dir, const fs::path& output_dir) {
  fs::path output_dir_ = output_dir;
  if (output_dir_.empty()) output_dir_ = "./pathfinder-tf";

  TFPOVGenerator generator;
  std::vector<std::pair<std::string, std::string>> output;

  for (const auto& entry : fs::directory_iterator(buggy_input_dir)) {
    auto path = entry.path();
    if (fs::is_directory(path)) {
      std::string api_name = path.filename();
      size_t idx = 1;
      for (const auto& entry : fs::directory_iterator(path)) {
        auto seed_path = entry.path();
        if (!fs::is_regular_file(seed_path)) continue;
        std::vector<long> concrete_args =
            uint8_vec_to_long_vec(file_to_vector(seed_path));
        std::string target_name;
        if (idx == 1) {
          target_name = api_name;
        } else {
          target_name = api_name + "_" + std::to_string(idx);
        }
        std::unique_ptr<TFAPI> api = get_api(api_name);
        api->assign(concrete_args);
        std::string code = generator.gen(api.get());
        output.push_back({target_name, code});
        idx++;
      }
    }
  }

  std::string template_dir_name = "tf" + DLL_VERSION;
  fs::path tf_template_dir_path = template_dir_path() / template_dir_name;
  cp_dir_recursive(tf_template_dir_path, output_dir_);
  fs::path pov_dir(output_dir_ / "pov");
  create_directories(pov_dir);

  std::string bazel_contents =
      "load(\"//tensorflow:tensorflow.bzl\", \"tf_cc_binary\")\n\n"

      "package(\n"
      "    default_visibility = [\"//visibility:public\"],\n"
      ")\n\n";

  for (auto& p : output) {
    std::string target_name = p.first;
    std::string code = p.second;
    fs::path file_path = pov_dir / (target_name + ".cc");
    write_file(file_path.string(), code);

    bazel_contents +=
        "tf_cc_binary(\n"
        "    name = \"pov_" +
        target_name +
        "\",\n"
        "    srcs = [\n"
        "        \"" +
        target_name +
        ".cc\",\n"
        "    ],\n"
        "    deps = [\n"
        "        \"//tensorflow/core/kernels/pathfinder:fuzzer_util\",\n"
        "        \"//tensorflow/cc:scope\",\n"
        "        \"//tensorflow/core:core_cpu\",\n"
        "        \"//tensorflow/core:tensorflow\",\n"
        "        \"//tensorflow/cc:cc_ops\",\n"
        "    ],\n"
        ")\n\n";
  }

  fs::path bazel_file_path = pov_dir / "BUILD";
  write_file(bazel_file_path.string(), bazel_contents);
}
