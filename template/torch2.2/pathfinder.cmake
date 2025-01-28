add_library(fuzzer_util SHARED fuzzer_util.cpp)
target_link_libraries(fuzzer_util PRIVATE ${TORCH_LIBRARIES})

# Fuzz driver
function(add_pathfinder_fuzz_driver fuzz_driver_name)
  set(fuzz_driver_src ${fuzz_driver_name}.cpp)
  set(fuzz_driver_exe pathfinder_fuzz_driver_${fuzz_driver_name})

  add_executable(${fuzz_driver_exe} ${fuzz_driver_src})
  target_include_directories(${fuzz_driver_exe} PRIVATE "${PROJECT_SOURCE_DIR}")


  target_link_libraries(${fuzz_driver_exe} PRIVATE
    pathfinder)
  target_link_libraries(${fuzz_driver_exe} PRIVATE
    ${TORCH_LIBRARIES}
    fuzzer_util)
  if(USE_CPP_CODE_COVERAGE)
    target_link_options(${fuzz_driver_exe} PRIVATE --coverage)
  endif()

endfunction()

# PoV
function(add_pov pov_name)
  set(pov_src ${pov_name}.cpp)
  set(pov_exe pov_${pov_name})

  add_executable(${pov_exe} ${pov_src})
  target_include_directories(${pov_exe} PRIVATE "${PROJECT_SOURCE_DIR}")

  target_link_libraries(${pov_exe} PRIVATE
    ${TORCH_LIBRARIES}
    fuzzer_util)

endfunction()
