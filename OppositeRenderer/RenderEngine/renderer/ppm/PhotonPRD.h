/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "renderer/RandomState.h"
struct PhotonPRD
{
    optix::float3 power;
    float weight;
    optix::uint pm_index;
    optix::uint numStoredPhotons;
    optix::uint depth;
    RandomState randomState;
};
