#include "writer.h"

#include "options.h"

Writer::Writer(std::string dll_, std::string mode_, fs::path output_dir_)
    : dll(dll_), mode(mode_), output_dir(output_dir_) {
  PDG_ASSERT(dll == "torch" || dll == "tf");
  PDG_ASSERT(mode == "driver" || mode == "pov");

  if (output_dir.empty()) output_dir = "./pathfinder-" + dll;
}
void Writer::add(std::string file_name, std::string code) {
  output.push_back({file_name, code});
}
void Writer::write() const {
  copy_mold();

  fs::path target_dir_path = output_dir / mode;

  std::string cmake_func_name;
  if (mode == "driver") {
    cmake_func_name = "add_pathfinder_fuzz_driver";
  } else if (mode == "pov") {
    cmake_func_name = "add_pov";
  }

  std::set<std::string> names_seen;
  create_directories(target_dir_path);
  std::string cmake_contents;
  for (auto& p : output) {
    std::string file_name = p.first + ".cpp";
    auto code = p.second;
    if ((names_seen.find(file_name) != names_seen.end())) continue;
    names_seen.insert(file_name);
    fs::path file_path = target_dir_path / file_name;
    write_file(file_path, code);
    cmake_contents += cmake_func_name + "(" + strip_ext(file_name) + ")\n";
  }
  fs::path cmake_file_path = target_dir_path / "CMakeLists.txt";
  write_file(cmake_file_path, cmake_contents);
}

void Writer::copy_mold() const {
  std::string template_dir_name = dll + DLL_VERSION;
  fs::path mold_dir = template_dir_path() / template_dir_name;
  cp_dir_recursive(mold_dir, output_dir);
}
