#include "options.h"

#include <getopt.h>

#include <cassert>
#include <climits>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
namespace fs = std::filesystem;

enum PDG_OPTION {
  OPT_DLL,
  OPT_DLL_VERSION,
  OPT_MODE,
  OPT_WO_STAGED,
  OPT_BUGGY_INPUT_DIR,
  OPT_OUTPUT,

  OPT_HELP,
};

option longopts[] = {
    {"dll", required_argument, NULL, OPT_DLL},
    {"dll_version", required_argument, NULL, OPT_DLL_VERSION},
    {"mode", required_argument, NULL, OPT_MODE},
    {"wo_staged", no_argument, NULL, OPT_WO_STAGED},
    {"buggy_input_dir", required_argument, NULL, OPT_BUGGY_INPUT_DIR},
    {"output", required_argument, NULL, OPT_OUTPUT},

    {"help", no_argument, NULL, OPT_HELP},
    {0}};

std::string DLL;
std::string DLL_VERSION;
std::string MODE = "driver";
bool WO_STAGED = false;
fs::path BUGGY_INPUT_DIR_PATH;
fs::path OUTPUT_DIR_PATH;

void print_usage(int exit_code, char* program_name) {
  printf("Usage : %s [...]\n", program_name);
  printf(
      "    --dll                  Target deep learning library. Should be one "
      "of {torch, tf}.\n"
      "    --dll_version          Version of DL library.\n"
      "    --mode                 Gen mode. Should be one of {driver, pov}. "
      "(default=driver)\n"
      "    --wo_staged            For ablation study. Turn off staged "
      "synthesis.\n"
      "    --buggy_input_dir      Path to buggy input dir (PoV mode only).\n"
      "    --output               Output dir path. If not specified, set to "
      "cwd.\n\n"

      "    --help                 Display this usage information.\n");
  exit(exit_code);
}

void parse_arg(int argc, char** argv) {
  while (1) {
    int opt = getopt_long(argc, argv, "", longopts, 0);
    if (opt == -1) break;
    switch (opt) {
      case OPT_DLL:
        DLL = optarg;
        break;
      case OPT_DLL_VERSION:
        DLL_VERSION = optarg;
        break;
      case OPT_MODE:
        MODE = optarg;
        break;
      case OPT_WO_STAGED:
        WO_STAGED = true;
        break;
      case OPT_BUGGY_INPUT_DIR:
        BUGGY_INPUT_DIR_PATH = optarg;
        break;
      case OPT_OUTPUT:
        OUTPUT_DIR_PATH = optarg;
        break;
      case OPT_HELP:
      case '?':
        print_usage(1, argv[0]);
        break;
      default:
        assert(false);
    }
  }
}
