#!/bin/bash
#
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
#
SOURCE_DIR=$(pwd)/cpp
BUILD_PATH=$(pwd)/build
CMAKE_FLAGS=""

print_line() {
echo "=================================================================";
}

print_line
echo "Building Simple-CuDF project"
print_line

echo "SOURCE_DIR: ${SOURCE_DIR}"
echo "BUILD_PATH: ${BUILD_PATH}"
mkdir -p ${BUILD_PATH}
pushd ${BUILD_PATH} || exit 1
cmake ${SOURCE_DIR} || exit 1
make -j 4 || exit 1
printf "GCylon CPP Built Successfully!"
popd || exit 1
print_line

