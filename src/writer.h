#ifndef PATHFINDER_DRIVER_GENERATOR_WRITER
#define PATHFINDER_DRIVER_GENERATOR_WRITER

#include <cassert>
#include <iostream>

#include "utils.h"

class Writer {
  typedef std::pair<std::string, std::string> api;

 public:
  Writer(std::string dll_, std::string mode_, fs::path output_dir_);
  void add(std::string file_name, std::string code);
  void write() const;

 private:
  std::string dll;
  std::string mode;
  fs::path output_dir;
  std::vector<api> output;

  void copy_mold() const;
};

#endif
