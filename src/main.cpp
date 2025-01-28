#include <cassert>
#include <filesystem>
#include <iostream>

#include "options.h"
#include "tf_gen.h"
#include "torch_gen.h"
#include "utils.h"
namespace fs = std::filesystem;

int main(int argc, char **argv) {
  parse_arg(argc, argv);

  PDG_CHECK(DLL == "torch" || DLL == "tf",
            "Invalid dll `" << DLL << "`. Should be one of {torch, tf}.");
  if (DLL == "torch") {
    PDG_CHECK(DLL_VERSION == "1.11" || DLL_VERSION == "2.2",
              "Supporte PyTorch versions: {1.11, 2.2}")
  } else if (DLL == "tf") {
    PDG_CHECK(DLL_VERSION == "2.16",
              "Supporte TensorFlow version: 2.16")
  }
  PDG_CHECK(MODE == "driver" || MODE == "pov",
            "Invalid mode `" + MODE + "`. Should be one of {driver, pov}.");

  if (MODE == "driver") {
    if (DLL == "torch") {
      gen_torch_drivers(OUTPUT_DIR_PATH);
    } else if (DLL == "tf") {
      gen_tf_drivers(OUTPUT_DIR_PATH);
    }
  } else if (MODE == "pov") {
    PDG_CHECK(!BUGGY_INPUT_DIR_PATH.empty(), "Buggy input dir is not set");
    if (DLL == "torch") {
      gen_torch_pov(BUGGY_INPUT_DIR_PATH, OUTPUT_DIR_PATH);
    } else if (DLL == "tf") {
      gen_tf_pov(BUGGY_INPUT_DIR_PATH, OUTPUT_DIR_PATH);
    }
  }

  return 0;
}
