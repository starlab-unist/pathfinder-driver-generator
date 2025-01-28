#ifndef PATHFINDER_DRIVER_GENERATOR_UTILS
#define PATHFINDER_DRIVER_GENERATOR_UTILS

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
namespace fs = std::filesystem;

#define PDG_CHECK(cond, msg)                                                 \
  if (!(cond)) {                                                             \
    std::cerr << "PDG CHECK FAILED: " << msg << std::endl;                   \
    std::cerr << "at `" << __PRETTY_FUNCTION__ << "` in " << __FILE__ << ":" \
              << __LINE__ << std::endl;                                      \
    exit(0);                                                                 \
  }

#define PDG_ASSERT(cond)                                                     \
  if (!(cond)) {                                                             \
    std::cerr << "PDG ASSERTION FAILED: `" << __PRETTY_FUNCTION__ << "` in " \
              << __FILE__ << ":" << __LINE__ << std::endl;                   \
    exit(1);                                                                 \
  }

#define NOT_IMPLEMENTED                                                 \
  {                                                                     \
    std::cerr << "NOT IMPLEMENTED: `" << __PRETTY_FUNCTION__ << "` in " \
              << __FILE__ << ":" << __LINE__ << std::endl;              \
    exit(1);                                                            \
  }

class Unreachable : public std::exception {};
#define UNREACHABLE throw Unreachable()

static const std::string space = " ";
static const std::string lte = " <= ";
static const std::string gte = " >= ";
static const std::string assign = " = ";
static const std::string comma = ", ";
static const std::string semicolon = ";";
static const std::string newline = "\n";

bool startswith(const std::string& base, std::string prefix);
bool endswith(const std::string& base, std::string suffix);
std::string strip(const std::string& str);
std::string strip_ext(std::string filename);
std::string& strip_prefix(std::string& orig, std::string prefix);
std::string& strip_suffix(std::string& orig, std::string suffix);
std::vector<std::string> split(const std::string& orig, std::string sep);
std::pair<std::string, std::string> lsplit(const std::string& orig,
                                           std::string sep);
std::string unique_name(std::string name, std::set<std::string>& names_seen);
std::string snake_to_camel(const std::string& orig);

std::string quoted(std::string param_name);
std::string sq_quoted(std::string param_name);
std::string bracket(std::string str = "");
std::string square(std::string str);
std::string curly(std::string str);
std::string join_strs(const std::vector<std::string>& strs,
                      std::string sep = comma);
std::string join_quoted_strs(const std::vector<std::string>& strs,
                             std::string sep = comma);
void concat(std::vector<std::string>& left, const std::string& prefix,
            const std::vector<std::string>& right,
            const std::string& postfix = "");
void tie_strs(std::vector<std::string>& strs);

void write_file(const fs::path& file_path, std::string contents);
fs::path api_signatures_dir_path();
fs::path template_dir_path();
void cp_dir_recursive(const fs::path& from, const fs::path& to);
std::vector<uint8_t> file_to_vector(const fs::path& filepath);
std::vector<long> uint8_vec_to_long_vec(std::vector<uint8_t> uint8_vec);

template <typename T>
void concat(std::vector<T>& left, const std::vector<T>& right) {
  for (auto& elem : right) left.push_back(elem);
}

template <typename T>
T& head(std::vector<T>& vec) {
  return vec.at(0);
}

template <typename T>
std::vector<T> tail(std::vector<T> orig) {
  typename std::vector<T>::const_iterator first = orig.begin() + 1;
  typename std::vector<T>::const_iterator last = orig.end();
  return std::vector<T>(first, last);
}

template <typename T>
T pop_front(std::vector<T>& vec) {
  PDG_CHECK(!vec.empty(), "Pop from empty vector");
  T head = vec.front();
  typename std::vector<T>::const_iterator first = vec.begin() + 1;
  typename std::vector<T>::const_iterator last = vec.end();
  std::vector<T> tail(first, last);
  vec = tail;
  return head;
}

#endif
