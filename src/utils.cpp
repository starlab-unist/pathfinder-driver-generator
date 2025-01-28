#include "utils.h"

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
namespace fs = std::filesystem;

std::string strip_ext(std::string filename) {
  return filename.substr(0, filename.find_last_of("."));
}

bool startswith(const std::string& base, std::string prefix) {
  return base.compare(0, prefix.size(), prefix) == 0;
}

bool endswith(const std::string& base, std::string suffix) {
  return base.length() >= suffix.length() &&
         base.compare(base.length() - suffix.length(), suffix.length(),
                      suffix) == 0;
}

bool is_space(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

std::string strip(const std::string& str) {
  if (str.size() == 0) return str;

  size_t pos_right = str.size() - 1;
  for (; pos_right > 0; pos_right--)
    if (!is_space(str[pos_right])) break;

  size_t pos_left = 0;
  for (; pos_left <= pos_right; pos_left++)
    if (!is_space(str[pos_left])) break;

  return str.substr(pos_left, pos_right - pos_left + 1);
}

std::string& strip_prefix(std::string& orig, std::string prefix) {
  if (startswith(orig, prefix)) orig = orig.substr(prefix.size());
  return orig;
}

std::string& strip_suffix(std::string& orig, std::string suffix) {
  if (endswith(orig, suffix))
    orig = orig.substr(0, orig.size() - suffix.size());
  return orig;
}

std::vector<std::string> split(const std::string& orig, std::string sep) {
  std::vector<std::string> chunks;
  size_t pos = 0;
  while (true) {
    if (pos > orig.size()) break;

    size_t found_pos = orig.find(sep, pos);
    if (found_pos == std::string::npos) {
      chunks.push_back(orig.substr(pos));
      break;
    }

    chunks.push_back(orig.substr(pos, found_pos - pos));
    pos = found_pos + sep.size();
  }
  return chunks;
}

std::pair<std::string, std::string> lsplit(const std::string& orig,
                                           std::string sep) {
  std::string left;
  std::string right;

  size_t found_pos = orig.find(sep, 0);
  if (found_pos == std::string::npos) return {orig, ""};
  return {orig.substr(0, found_pos), orig.substr(found_pos + sep.size())};
}

std::string unique_name(std::string name, std::set<std::string>& names_seen) {
  std::string unique_name = name;
  size_t id = 0;
  while (names_seen.find(unique_name) != names_seen.end())
    unique_name = name + "_" + std::to_string(id++);
  names_seen.insert(unique_name);
  return unique_name;
}

std::string snake_to_camel(const std::string& orig) {
  std::vector<std::string> words = split(orig, "_");
  std::string cameled;
  for (auto& word : words) {
    word = strip(word);
    if (!word.empty()) word[0] = toupper(word[0]);
    cameled += word;
  }
  return cameled;
}

std::string quoted(std::string param_name) { return "\"" + param_name + "\""; }

std::string sq_quoted(std::string param_name) {
  return "[" + quoted(param_name) + "]";
}

std::string bracket(std::string str) { return "(" + str + ")"; }

std::string square(std::string str) { return "[" + str + "]"; }

std::string curly(std::string str) { return "{" + str + "}"; }

std::string join_strs(const std::vector<std::string>& strs, std::string sep) {
  std::string joined;
  for (size_t i = 0; i < strs.size(); i++) {
    joined += strs[i];
    if (i != strs.size() - 1) joined += sep;
  }
  return joined;
}

std::string join_quoted_strs(const std::vector<std::string>& strs,
                             std::string sep) {
  std::string joined;
  for (size_t i = 0; i < strs.size(); i++) {
    joined += quoted(strs[i]);
    if (i != strs.size() - 1) joined += sep;
  }
  return joined;
}

void concat(std::vector<std::string>& left, const std::string& prefix,
            const std::vector<std::string>& right, const std::string& postfix) {
  for (auto& elem : right) left.push_back(prefix + elem + postfix);
}

void tie_strs(std::vector<std::string>& strs) {
  for (size_t i = 0; i < strs.size(); i++)
    if (i != strs.size() - 1) strip_suffix(strs[i], "\n");
}

void write_file(const fs::path& file_path, std::string contents) {
  std::ofstream f(file_path);
  if (f.is_open()) {
    f << contents;
    f.close();
  } else {
    assert(false);
  }
}

fs::path api_signatures_dir_path() {
  fs::path current_file_path(__FILE__);
  fs::path src_dir_path = current_file_path.parent_path();
  fs::path pdg_home_path = src_dir_path.parent_path();
  return pdg_home_path / "api-signatures";
}

fs::path template_dir_path() {
  fs::path current_file_path(__FILE__);
  fs::path src_dir_path = current_file_path.parent_path();
  fs::path pdg_home_path = src_dir_path.parent_path();
  return pdg_home_path / "template";
}

void cp_dir_recursive(const fs::path& from, const fs::path& to) {
  try {
    fs::copy(
        from, to,
        fs::copy_options::overwrite_existing | fs::copy_options::recursive);
  } catch (std::exception& e) {
    std::cerr << e.what();
    exit(1);
  }
}

std::vector<uint8_t> file_to_vector(const fs::path& filepath) {
  // from
  // https://www.coniferproductions.com/posts/2022/10/25/reading-binary-files-cpp/

  std::ifstream inputFile(filepath, std::ios_base::binary);

  inputFile.seekg(0, std::ios_base::end);
  auto length = inputFile.tellg();
  inputFile.seekg(0, std::ios_base::beg);

  std::vector<uint8_t> buffer(length);
  inputFile.read(reinterpret_cast<char*>(buffer.data()), length);

  inputFile.close();
  return buffer;
}

std::vector<long> uint8_vec_to_long_vec(std::vector<uint8_t> uint8_vec) {
  size_t unit = sizeof(long) / sizeof(uint8_t);
  size_t size = uint8_vec.size() / unit;
  std::vector<long> long_vec(size);
  memcpy(long_vec.data(), uint8_vec.data(), sizeof(long) * size);
  assert(long_vec.size() == size);
  return long_vec;
}
