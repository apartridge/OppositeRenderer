/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "RandomState.h"

struct RadiancePRD
{
    optix::float3 attenuation;
    optix::float3 radiance;
    optix::uint depth;
    optix::float3 position;
    optix::float3 normal;
    optix::uint flags;
    union
    {
        optix::float3 randomNewDirection;
        optix::float3 Le;
    };
    RandomState randomState;
#if ENABLE_PARTICIPATING_MEDIA
    optix::float3 volumetricRadiance;
#endif
    float lastTHit;
};

#define PRD_HIT_EMITTER (1<<31u)
#define PRD_ERROR (1<<30u)
#define PRD_MISS (1<<29u)
#define PRD_HIT_SPECULAR (1<<28u)
#define PRD_HIT_NON_SPECULAR (1<<27u)
#define PRD_PATH_TRACING (1<<26u)