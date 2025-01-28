cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(
  PathFinderDriverGenerator
  VERSION 1.0
  LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type not set, default is Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build from: Debug Release." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(PDGInclude
  "${PROJECT_SOURCE_DIR}")

add_subdirectory(json)

add_subdirectory(src)
