#ifndef PATHFINDER_DRIVER_GENERATOR_TORCH_GEN
#define PATHFINDER_DRIVER_GENERATOR_TORCH_GEN

#include <filesystem>
#include <vector>
namespace fs = std::filesystem;

void gen_torch_drivers(const fs::path& output_dir);
void gen_torch_pov(const fs::path& buggy_input_dir, const fs::path& output_dir);

#endif
