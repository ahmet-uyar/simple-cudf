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

if (NOT DEFINED ENV{CONDA_PREFIX})
    message( FATAL_ERROR "ENV{CONDA_PREFIX} not defined. It has to be defined.")
endif ()

set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
set(CONDA_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
set(CONDA_LINK_DIRS "$ENV{CONDA_PREFIX}/lib")
set(CUDF_CUDA_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include/libcudf/libcudacxx/")
message(STATUS "Conda environment detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

message("CUDAToolkit_INCLUDE_DIRS: " ${CUDAToolkit_INCLUDE_DIRS})
message("CONDA_INCLUDE_DIRS: " ${CONDA_INCLUDE_DIRS})
message("CUDF_CUDA_INCLUDE_DIR: " ${CUDF_CUDA_INCLUDE_DIR})

message("CONDA_LIB_DIR: " ${CONDA_LINK_DIRS})
message("CUDA_LIB_DIR: " ${CUDAToolkit_LIBRARY_DIR})

include_directories(../cpp
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CUDF_CUDA_INCLUDE_DIR}
        ${CONDA_INCLUDE_DIRS}
        )

link_directories(
        ${CUDAToolkit_LIBRARY_DIR}
        ${CONDA_LINK_DIRS}
        )

#build a binary with cuda
macro(add_exe EXENAME)
    add_executable(${EXENAME} ${EXENAME}.cpp construct.h)
    target_link_libraries(${EXENAME} cudf CUDA::cudart)
endmacro(add_exe)

#build a binary with cuda
macro(add_cuda_exe EXENAME)
    add_executable(${EXENAME} ${EXENAME}.cu)
    target_link_libraries(${EXENAME} CUDA::cudart)
endmacro(add_cuda_exe)

add_exe(read_table)
add_exe(minmax)
#add_cuda_exe(rand)
add_exe(sorting)
