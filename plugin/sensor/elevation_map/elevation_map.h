// Copyright 2023 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_PLUGIN_SENSOR_ELEVATION_MAP_H_
#define MUJOCO_PLUGIN_SENSOR_ELEVATION_MAP_H_

#include <optional>
#include <vector>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mjvisualize.h>

namespace mujoco::plugin::sensor
{

// A elevation map outputs elevation gridmap representing height of environment.
//
// The sensor parameters:
//  1. (int) Horizontal dimension (dim_x).
//  2. (int) Vertical dimension (dim_y).
//  3. (float) Cell resolution, in meter.
//  4. (float) Maximum height limit, in meter.
//  5. (int) Skip count. Elevation map is updated(calculated) once a 'skip_count'.
class ElevationMap
{
public:
    static ElevationMap *Create(const mjModel *m, mjData *d, int instance);
    ElevationMap(ElevationMap &&) = default;
    ~ElevationMap() = default;

    void Reset(const mjModel *m, int instance);
    void Compute(const mjModel *m, mjData *d, int instance);
    void Visualize(const mjModel *m, mjData *d, const mjvOption *opt,
                   mjvScene *scn, int instance);

    static void RegisterPlugin();

    int size_[2]; // horizontal and vertical resolution
    mjtNum res_;
    mjtNum max_height_;
    int skip_count_;

private:
    ElevationMap(const mjModel *m, mjData *d, int instance, int *size, mjtNum res, mjtNum max_h, int skip_count);
    mjtNum map_offset_[2];
    std::vector<mjtNum> elevation_;
    std::vector<mjtNum> origin_xy_;
    int compute_count_{0};
};

} // namespace mujoco::plugin::sensor

#endif // MUJOCO_PLUGIN_SENSOR_ELEVATION_MAP_H_
