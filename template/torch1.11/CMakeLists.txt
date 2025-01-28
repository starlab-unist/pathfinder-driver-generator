cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(
  PathFinderTorch
  VERSION 1.0
  LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type not set, default is Debug")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build from: Debug Release." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(GEN_DRIVER)
  find_package(Z3)
  message(STATUS "Z3_FOUND: ${Z3_FOUND}")
  message(STATUS "Found Z3 ${Z3_VERSION_STRING}")
  message(STATUS "Z3_DIR: ${Z3_DIR}")

  find_package(pathfinder REQUIRED)
  message(STATUS "PATHFINDER_FOUND: ${PATHFINDER_FOUND}")
  message(STATUS "Found PATHFINDER ${PATHFINDER_VERSION_STRING}")
  message(STATUS "PATHFINDER_DIR: ${PATHFINDER_DIR}")

  include(pathfinder.cmake)
  add_subdirectory(driver)
elseif(GEN_POV)
  include(pathfinder.cmake)
  add_subdirectory(pov)
endif()
