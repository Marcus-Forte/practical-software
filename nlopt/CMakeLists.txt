cmake_minimum_required(VERSION 3.15)
project(nlopt)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(matplotlib_cpp)
if(matplotlib_cpp_FOUND)
  message(STATUS "MATPLOTLIB found!")
  link_libraries(matplotlib_cpp::matplotlib_cpp)
  add_definitions(-DUSE_MATPLOTLIB)
endif()

find_package(Eigen3 REQUIRED)
add_executable(nlopt src/nlopt.cc)
target_link_libraries(nlopt Eigen3::Eigen)

