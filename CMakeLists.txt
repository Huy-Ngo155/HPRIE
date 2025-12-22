cmake_minimum_required(VERSION 3.20)
project(HPRIE VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_library(NUMA_LIB numa REQUIRED)

add_executable(HPRIE main.cpp)
target_link_libraries(HPRIE PRIVATE ${NUMA_LIB} pthread)
target_compile_options(HPRIE PRIVATE -O3 -march=native)
