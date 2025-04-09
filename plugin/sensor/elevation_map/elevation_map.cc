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

#include "elevation_map.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjplugin.h>
#include <mujoco/mjtnum.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mujoco.h>

namespace mujoco::plugin::sensor
{

namespace
{

// Checks that a plugin config attribute exists.
bool CheckAttr(const std::string &input)
{
    char *end;
    std::string value = input;
    value.erase(std::remove_if(value.begin(), value.end(), isspace), value.end());
    strtod(value.c_str(), &end);
    return end == value.data() + value.size();
}

// Converts a string into a numeric vector
template <typename T>
void ReadVector(std::vector<T> &output, const std::string &input)
{
    std::stringstream ss(input);
    std::string item;
    char delim = ' ';
    while (getline(ss, item, delim))
    {
        CheckAttr(item);
        output.push_back(strtod(item.c_str(), nullptr));
    }
}

} // namespace

// Creates a ElevationMap instance if all config attributes are defined and
// within their allowed bounds.
ElevationMap *ElevationMap::Create(const mjModel *m, mjData *d,
                                   int instance)
{
    if (CheckAttr(std::string(mj_getPluginConfig(m, instance, "res"))) &&
        CheckAttr(std::string(mj_getPluginConfig(m, instance, "dim"))) &&
        CheckAttr(std::string(mj_getPluginConfig(m, instance, "max_height"))) &&
        CheckAttr(std::string(mj_getPluginConfig(m, instance, "skip_count"))))
    {
        // dimension
        std::vector<int> size;
        std::string size_str = std::string(mj_getPluginConfig(m, instance, "dim"));
        ReadVector(size, size_str.c_str());
        if (size.size() != 2)
        {
            mju_error("Both horizontal and vertical resolutions must be specified");
            return nullptr;
        }
        if (size[0] <= 0 || size[1] <= 0)
        {
            mju_error("Horizontal and vertical resolutions must be positive");
            return nullptr;
        }

        // resolution
        mjtNum res = strtod(mj_getPluginConfig(m, instance, "res"), nullptr);
        if (res < 0 || res > 1)
        {
            mju_error("`res` must be a nonnegative float between [0, 1]");
            return nullptr;
        }

        // maximum height
        mjtNum max_h = strtod(mj_getPluginConfig(m, instance, "max_height"), nullptr);
        if (max_h < 0)
        {
            mju_error("`max_height` must be a nonnegative float");
            return nullptr;
        }

        // skip count
        int skip_count = strtod(mj_getPluginConfig(m, instance, "skip_count"), nullptr);
        if (skip_count < 0)
        {
            mju_error("`skip_count` must be a nonnegative integer");
            return nullptr;
        }

        return new ElevationMap(m, d, instance, size.data(), res, max_h, skip_count);
    }
    else
    {
        mju_error("Invalid or missing parameters in elevation_map sensor plugin");
        return nullptr;
    }
}

ElevationMap::ElevationMap(const mjModel *m, mjData *d, int instance, int *size, mjtNum res, mjtNum max_h, int skip_count)
    : size_{size[0], size[1]},
      res_(res),
      max_height_(max_h),
      skip_count_(skip_count)
{
    // Make sure sensor is attached to a site.
    for (int i = 0; i < m->nsensor; ++i)
    {
        if (m->sensor_type[i] == mjSENS_PLUGIN && m->sensor_plugin[i] == instance)
        {
            if (m->sensor_objtype[i] != mjOBJ_SITE)
            {
                mju_error("Elevation map sensor must be attached to a site");
            }
        }
    }

    // Allocate distance array.
    elevation_.resize(size_[0] * size_[1], 0);
    // Allocate and calculate ray-origin coordinate(x, y).
    origin_xy_.resize(size_[0] * size_[1] * 2, 0);
    for (int i = 0; i < (size_[0] * size_[1]); i++)
    {
        origin_xy_.data()[2 * i + 0] = (int)(i / size_[1]) * res_ + 0.5 * res_;
        origin_xy_.data()[2 * i + 1] = (int)(i % size_[1]) * res_ + 0.5 * res_;
    }
}

void ElevationMap::Reset(const mjModel *m, int instance) {}

void ElevationMap::Compute(const mjModel *m, mjData *d, int instance)
{
    mj_markStack(d);

    // Check compute count
    compute_count_--;
    if (compute_count_ > 0)
    {
        mj_freeStack(d);
        return;
    }
    else
        compute_count_ = skip_count_;

    // Get sensor id.
    int id;
    for (id = 0; id < m->nsensor; ++id)
    {
        if (m->sensor_type[id] == mjSENS_PLUGIN &&
            m->sensor_plugin[id] == instance)
        {
            break;
        }
    }

    // Clear sensordata and distance matrix.
    mjtNum *sensordata = d->sensordata + m->sensor_adr[id];
    mju_zero(sensordata, m->sensor_dim[id]);
    int frame = size_[0] * size_[1];
    mju_zero(elevation_.data(), frame);

    // Get site id.
    int site_id = m->sensor_objid[id];

    // Get site frame.
    mjtNum *site_pos = d->site_xpos + 3 * site_id;
    mjtNum *site_mat = d->site_xmat + 9 * site_id;

    map_offset_[0] = (mjtNum)((int)(site_pos[0] / res_) * res_) - 0.5 * size_[0] * res_;
    map_offset_[1] = (mjtNum)((int)(site_pos[1] / res_) * res_) - 0.5 * size_[1] * res_;

    // Compute distance.
    for (int i = 0; i < size_[0]; i++)
    {
        for (int j = 0; j < size_[1]; j++)
        {
            mjtNum origin[3];
            origin[0] = map_offset_[0] + origin_xy_.data()[2 * (size_[1] * i + j) + 0];
            origin[1] = map_offset_[1] + origin_xy_.data()[2 * (size_[1] * i + j) + 1];
            origin[2] = max_height_;

            int geomid[1];
            mjtByte geomgroup[mjNGROUP] = {1, 0, 0, 0, 0, 0};
            mjtNum vec[3]{0.0, 0.0, -1.0};
            mjtNum dist = mj_ray(m, d, origin, vec, geomgroup, 1, -1, geomid);
            if (dist < 0)
                elevation_.data()[i * size_[1] + j] = NAN;
            else
                elevation_.data()[i * size_[1] + j] = origin[2] - dist;
        }
    }

    mj_freeStack(d);
}

// Thickness of taxel-visualization boxes relative to contact distance.
static const mjtNum kRelativeThickness[3]{0.9, 0.9, 0.02};

void ElevationMap::Visualize(const mjModel *m, mjData *d, const mjvOption *opt,
                             mjvScene *scn, int instance)
{
    mj_markStack(d);

    // Get sensor id.
    int id;
    for (id = 0; id < m->nsensor; ++id)
    {
        if (m->sensor_type[id] == mjSENS_PLUGIN &&
            m->sensor_plugin[id] == instance)
        {
            break;
        }
    }

    // Get sensor data.
    mjtNum *sensordata = d->sensordata + m->sensor_adr[id];

    // Get site id and frame.
    int site_id = m->sensor_objid[id];
    mjtNum *site_pos = d->site_xpos + 3 * site_id;
    mjtNum *site_mat = d->site_xmat + 9 * site_id;
    // mjtNum site_quat[4];
    // mju_mat2Quat(site_quat, site_mat);

    // Draw geoms.
    for (int i = 0; i < size_[0]; i++)
    {
        for (int j = 0; j < size_[1]; j++)
        {
            mjtNum elevation = elevation_.data()[i * size_[1] + j];
            if (std::isnan(elevation))
            {
                continue;
            }
            if (scn->ngeom >= scn->maxgeom)
            {
                mj_warning(d, mjWARN_VGEOMFULL, scn->maxgeom);
                mj_freeStack(d);
                return;
            }
            else
            {
                // size
                mjtNum size[3] = {0.5 * res_ * kRelativeThickness[0],
                                  0.5 * res_ * kRelativeThickness[1],
                                  0.5 * res_ * kRelativeThickness[2]};

                // position
                mjtNum pos[3];
                pos[0] = map_offset_[0] + origin_xy_.data()[2 * (size_[1] * i + j) + 0];
                pos[1] = map_offset_[1] + origin_xy_.data()[2 * (size_[1] * i + j) + 1];
                pos[2] = elevation - size[2];
                // mju_rotVecMat(pos, pos, site_mat);
                // mju_addTo3(pos, site_pos);

                // orientation
                mjtNum quat[4] = {1.0, 0.0, 0.0, 0.0};
                mjtNum mat[9];
                mju_quat2Mat(mat, quat);

                // color
                float rgba[4] = {0.0, 1.0, 0.0, 1.0};

                // draw box geom
                mjvGeom *thisgeom = scn->geoms + scn->ngeom;
                mjv_initGeom(thisgeom, mjGEOM_BOX, size, pos, mat, rgba);
                thisgeom->objtype = mjOBJ_UNKNOWN;
                thisgeom->objid = id;
                thisgeom->category = mjCAT_DECOR;
                thisgeom->segid = scn->ngeom;
                scn->ngeom++;
            }
        }
    }

    mj_freeStack(d);
}

void ElevationMap::RegisterPlugin()
{
    mjpPlugin plugin;
    mjp_defaultPlugin(&plugin);

    plugin.name = "mujoco.sensor.elevation_map";
    plugin.capabilityflags |= mjPLUGIN_SENSOR;

    // Parameterized by attributes.
    const char *attributes[] = {"dim", "res", "max_height", "skip_count"};
    plugin.nattribute = sizeof(attributes) / sizeof(attributes[0]);
    plugin.attributes = attributes;

    // Stateless.
    plugin.nstate = +[](const mjModel *m, int instance) { return 0; };

    // Sensor dimension = size[0] * size[1]
    plugin.nsensordata = +[](const mjModel *m, int instance, int sensor_id) {
        std::vector<int> size;
        std::string size_str = std::string(mj_getPluginConfig(m, instance, "dim"));
        ReadVector(size, size_str.c_str());
        return (2 + size[0] * size[1]); // map_offset(x,y) and elevation map data
    };

    // Can only run after positions have been updated.
    plugin.needstage = mjSTAGE_POS;

    // Initialization callback.
    plugin.init = +[](const mjModel *m, mjData *d, int instance) {
        auto *ElevationMap = ElevationMap::Create(m, d, instance);
        if (!ElevationMap)
        {
            return -1;
        }
        d->plugin_data[instance] = reinterpret_cast<uintptr_t>(ElevationMap);
        return 0;
    };

    // Destruction callback.
    plugin.destroy = +[](mjData *d, int instance) {
        delete reinterpret_cast<ElevationMap *>(d->plugin_data[instance]);
        d->plugin_data[instance] = 0;
    };

    // Reset callback.
    plugin.reset = +[](const mjModel *m, double *plugin_state, void *plugin_data,
                       int instance) {
        auto *ElevationMap = reinterpret_cast<class ElevationMap *>(plugin_data);
        ElevationMap->Reset(m, instance);
    };

    // Compute callback.
    plugin.compute =
        +[](const mjModel *m, mjData *d, int instance, int capability_bit) {
            auto *ElevationMap =
                reinterpret_cast<class ElevationMap *>(d->plugin_data[instance]);
            ElevationMap->Compute(m, d, instance);
        };

    // Visualization callback.
    plugin.visualize = +[](const mjModel *m, mjData *d, const mjvOption *opt,
                           mjvScene *scn, int instance) {
        auto *ElevationMap =
            reinterpret_cast<class ElevationMap *>(d->plugin_data[instance]);
        ElevationMap->Visualize(m, d, opt, scn, instance);
    };

    // Register the plugin.
    mjp_registerPlugin(&plugin);
}

} // namespace mujoco::plugin::sensor
