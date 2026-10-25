# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

cmake_minimum_required(VERSION 3.28)

# This forces Git to use the 'files' backend for all FetchContent operations.
# This fixes libcifpp and dssp incompatibility with newer git versions.
set(ENV{GIT_CONFIG_PARAMETERS} "'init.defaultRefFormat=files'")

# Resolve the active build Python before FetchContent configures native
# dependencies.  During a fresh conda env create this is the newly-created
# prefix, so its headers/libraries are discoverable without manual activation
# or a caller-provided CMAKE_PREFIX_PATH.
find_package(
  Python3
  COMPONENTS Interpreter Development NumPy
  REQUIRED)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import sys; print(sys.prefix)"
  OUTPUT_VARIABLE PYTHON_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
list(PREPEND CMAKE_PREFIX_PATH "${PYTHON_PREFIX}")

# Prefer the compiler shipped in the newly-created conda prefix when the
# caller did not provide one explicitly.  This keeps the editable build
# independent of the host compiler while still honoring CMAKE_CXX_COMPILER.
if(NOT CMAKE_CXX_COMPILER)
  find_program(
    CONDA_CXX_COMPILER
    NAMES x86_64-conda-linux-gnu-c++ c++ g++
    HINTS "${PYTHON_PREFIX}/bin"
    NO_DEFAULT_PATH)
  if(CONDA_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${CONDA_CXX_COMPILER}" CACHE FILEPATH
        "C++ compiler" FORCE)
  endif()
endif()

project(
  "${SKBUILD_PROJECT_NAME}"
  LANGUAGES CXX
  VERSION "${SKBUILD_PROJECT_VERSION}")

include(FetchContent)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(ABSL_PROPAGATE_CXX_STD ON)

# Remove support for scan deps, which is only useful when using C++ modules.
unset(CMAKE_CXX_SCANDEP_SOURCE)

# Commit-pinned archives avoid cloning complete upstream Git histories.
# SHA256 checks keep the source content fixed for fresh environment builds.
FetchContent_Declare(
  abseil-cpp
  URL https://codeload.github.com/abseil/abseil-cpp/tar.gz/d7aaad83b488fd62bd51c81ecf16cd938532cc0a
  URL_HASH SHA256=3a1f642ea21f45b220a01708d06b46042a3b750f31772b52371d6268f02d5bbc # 20240116.2
  EXCLUDE_FROM_ALL)

FetchContent_Declare(
  pybind11
  URL https://codeload.github.com/pybind/pybind11/tar.gz/2e0815278cb899b20870a67ca8205996ef47e70f
  URL_HASH SHA256=a50dfa85eda59b4c95180dd06baec89437ee1bc28ba92c5d9d1f853738ddd490 # v2.12.0
  EXCLUDE_FROM_ALL)

FetchContent_Declare(
  pybind11_abseil
  URL https://codeload.github.com/pybind/pybind11_abseil/tar.gz/bddf30141f9fec8e577f515313caec45f559d319
  URL_HASH SHA256=a92e5faba0397eb4a6ba79317ab57de7891dc69ca8848cf2326abe28d5179c8f # HEAD @ 2024-08-07
  EXCLUDE_FROM_ALL)

FetchContent_Declare(
  cifpp
  URL https://codeload.github.com/pdb-redo/libcifpp/tar.gz/ac98531a2fc8daf21131faa0c3d73766efa46180
  URL_HASH SHA256=5fa208efa5aa7890b2a31ac1743ffb4dfc8bdc2605f590cc4639cc8922b18890 # v7.0.3
  # Don't `EXCLUDE_FROM_ALL` as necessary for build_data.
)

FetchContent_Declare(
  dssp
  URL https://codeload.github.com/PDB-REDO/dssp/tar.gz/57560472b4260dc41f457706bc45fc6ef0bc0f10
  URL_HASH SHA256=6cf3068527117e9a613728633a62f489af8890d5c30e7db919183b470e972258 # v4.4.7
  EXCLUDE_FROM_ALL)

FetchContent_MakeAvailable(pybind11 abseil-cpp pybind11_abseil cifpp dssp)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(src/)

file(GLOB_RECURSE cpp_srcs src/alphafold3/*.cc)
list(FILTER cpp_srcs EXCLUDE REGEX ".*\(_test\|_main\|_benchmark\).cc$")

add_compile_definitions(NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

pybind11_add_module(cpp ${cpp_srcs})

target_link_libraries(
  cpp
  PRIVATE absl::check
          absl::flat_hash_map
          absl::node_hash_map
          absl::strings
          absl::status
          absl::statusor
          absl::log
          pybind11_abseil::absl_casters
          Python3::NumPy
          dssp::dssp
          cifpp::cifpp)

target_compile_definitions(cpp PRIVATE VERSION_INFO=${PROJECT_VERSION})
# The extension is installed at
# <prefix>/lib/python3.12/site-packages/alphafold3.  Resolve conda's shared
# libraries relative to that location so normal activation needs no
# LD_LIBRARY_PATH. Native extensions are built for the selected conda prefix;
# full conda-prefix relocation is a separate deployment concern.
set_target_properties(
  cpp
  PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../..")
install(TARGETS cpp LIBRARY DESTINATION alphafold3)
install(
  FILES LICENSE
        OUTPUT_TERMS_OF_USE.md
        WEIGHTS_PROHIBITED_USE_POLICY.md
        WEIGHTS_TERMS_OF_USE.md
  DESTINATION alphafold3)
