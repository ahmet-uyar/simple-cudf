##
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(SIMPLE_CUDF LANGUAGES CXX)

# C++ standard
# with cudf v21.08, at least c++ 17 required
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)
SET(CMAKE_REQUIRED_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-error=redundant-move")

# find CUDA
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 60 61 75)
enable_language(CUDA)

# if conda build is used
if("$ENV{CONDA_BUILD}" STREQUAL "1")
    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{BUILD_PREFIX};$ENV{PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{BUILD_PREFIX}/include" "$ENV{PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{BUILD_PREFIX}/lib" "$ENV{PREFIX}/lib")
    set(CMAKE_INSTALL_INCLUDEDIR "$ENV{BUILD_PREFIX}/include")
    set(CUDF_CUDA_INCLUDE_DIR "$ENV{BUILD_PREFIX}/include/libcudf/libcudacxx/" "$ENV{PREFIX}/include/libcudf/libcudacxx/")
    message(STATUS "Conda build detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
elseif(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{CONDA_PREFIX}/lib")
    set(CUDF_CUDA_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include/libcudf/libcudacxx/")
    message(STATUS "Conda environment detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
endif("$ENV{CONDA_BUILD}" STREQUAL "1")

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

message("CUDAToolkit_INCLUDE_DIRS: " ${CUDAToolkit_INCLUDE_DIRS})
message("CONDA_INCLUDE_DIRS: " ${CONDA_INCLUDE_DIRS})
message("CUDF_CUDA_INCLUDE_DIR: " ${CUDF_CUDA_INCLUDE_DIR})

include_directories(../cpp
        ${CUDF_CUDA_INCLUDE_DIR}
        ${CONDA_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
        )

link_directories("${CONDA_LINK_DIRS}")

#build a binary with cuda
macro(add_exe EXENAME)
    add_executable(${EXENAME} ${EXENAME}.cpp)
    target_link_libraries(${EXENAME} cudf CUDA::cudart)
endmacro(add_exe)

add_exe(read_table)
