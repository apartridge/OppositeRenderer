cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
message(STATUS "Using cmake ${CMAKE_VERSION} from ${CMAKE_COMMAND}")
project(OppositeRenderer LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

find_package(Qt5 REQUIRED COMPONENTS Widgets Gui Network)
find_package(OptiX REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)
find_package(assimp REQUIRED)

get_property(core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS)
string(REPLACE "-fPIC" "" new_core_options ${core_options})
set_property(TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options})
set_property(TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON")

add_subdirectory(Client)
add_subdirectory(Gui)
add_subdirectory(RenderEngine)
add_subdirectory(Server)
add_subdirectory(Standalone)
