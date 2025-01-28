#ifndef PATHFINDER_DRIVER_GENERATOR_OPTIONS
#define PATHFINDER_DRIVER_GENERATOR_OPTIONS

#include <filesystem>
#include <string>
namespace fs = std::filesystem;

extern std::string DLL;
extern std::string DLL_VERSION;
extern std::string MODE;
extern bool WO_STAGED;
extern fs::path BUGGY_INPUT_DIR_PATH;
extern fs::path OUTPUT_DIR_PATH;

void parse_arg(int argc, char** argv);

#endif
