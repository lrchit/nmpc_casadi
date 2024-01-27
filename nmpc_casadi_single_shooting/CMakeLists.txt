cmake_minimum_required(VERSION 3.16)
project(nmpc_caller)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

link_directories("./lib")
include_directories("./include")
add_executable(nmpc_caller
  src/nmpc_caller.cpp
)
target_link_libraries(nmpc_caller libyaml-cpp.a)
target_link_libraries(nmpc_caller /usr/local/lib/libcasadi.so.3.7)
