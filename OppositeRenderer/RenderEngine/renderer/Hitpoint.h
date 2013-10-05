/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#pragma once
#include "config.h"
#include "renderer/RadiancePRD.h"
struct Hitpoint
{
    optix::float3 position;
    optix::float3 normal;
    optix::float3 attenuation;
    optix::float3 radiance;
    optix::uint flags;
    optix::float3 volumetricRadiance;
};