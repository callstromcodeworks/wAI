cmake_minimum_required(VERSION 3.14)
project(fasttok CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(FASTTOK_NATIVE "Optimize for the build machine (-march=native)" ON)

find_package(Threads REQUIRED)

add_library(fasttok INTERFACE)
target_include_directories(fasttok INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(fasttok INTERFACE Threads::Threads)
if(FASTTOK_NATIVE AND NOT MSVC)
  target_compile_options(fasttok INTERFACE -march=native)
endif()
if(NOT MSVC)
  target_compile_options(fasttok INTERFACE $<$<CONFIG:Release>:-O3>)
endif()

add_executable(fasttok_cli src/fasttok_cli.cpp)
target_link_libraries(fasttok_cli PRIVATE fasttok)
set_target_properties(fasttok_cli PROPERTIES OUTPUT_NAME fasttok)

add_library(fasttok_c SHARED src/fasttok_capi.cpp)
target_link_libraries(fasttok_c PRIVATE fasttok)
set_target_properties(fasttok_c PROPERTIES OUTPUT_NAME fasttok)

add_executable(fasttok_unit tests/unit.cpp)
target_link_libraries(fasttok_unit PRIVATE fasttok)
