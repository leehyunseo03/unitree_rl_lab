cmake_minimum_required(VERSION 3.12)
project(g1_29dof_controller)

set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

set(UNITREE_SDK2_INCLUDE "$ENV{HOME}/.local/unitree_sdk2/include")
if(NOT EXISTS "${UNITREE_SDK2_INCLUDE}")
  set(UNITREE_SDK2_INCLUDE "${PROJECT_SOURCE_DIR}/../../../../.local/unitree_sdk2/include")
endif()
if(EXISTS "${UNITREE_SDK2_INCLUDE}")
  include_directories(${UNITREE_SDK2_INCLUDE})
  if(EXISTS "${UNITREE_SDK2_INCLUDE}/ddscxx")
    include_directories(${UNITREE_SDK2_INCLUDE}/ddscxx)
  endif()
endif()

include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx # dds
  /usr/local/include/iceoryx/v2.0.2 # dds
  include
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

link_directories(
  ${PROJECT_SOURCE_DIR}/../../../../.local/unitree_sdk2/lib
  /usr/lib/x86_64-linux-gnu
)

link_libraries(
  unitree_sdk2 ddsc ddscxx rt pthread # dds
  libboost_program_options.a libyaml-cpp.a fmt
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0
)

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(g1_ctrl main.cpp)