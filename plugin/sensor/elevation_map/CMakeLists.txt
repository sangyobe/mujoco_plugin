# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
find_package(mujoco CONFIG REQUIRED)

set(MUJOCO_SENSOR_INCLUDE
    ${ARTF_INSTALL_DIR}/include
)

set(MUJOCO_SENSOR_SRCS
    sensor.cc
    elevation_map.h
    elevation_map.cc
)

add_library(elevation_map SHARED)
target_sources(elevation_map PRIVATE ${MUJOCO_SENSOR_SRCS})
target_include_directories(elevation_map PRIVATE ${MUJOCO_SENSOR_INCLUDE})
target_link_libraries(elevation_map PRIVATE mujoco::mujoco)
target_compile_options(
    elevation_map
    PRIVATE ${AVX_COMPILE_OPTIONS}
    ${MUJOCO_MACOS_COMPILE_OPTIONS}
    ${EXTRA_COMPILE_OPTIONS}
    ${MUJOCO_CXX_FLAGS}
)
target_link_options(
    elevation_map
    PRIVATE
    ${MUJOCO_MACOS_LINK_OPTIONS}
    ${EXTRA_LINK_OPTIONS}
)
