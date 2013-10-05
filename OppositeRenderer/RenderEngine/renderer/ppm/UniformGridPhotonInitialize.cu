/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/
#include "config.h"

#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_STOCHASTIC_HASH
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "Photon.h"

using namespace optix;

rtBuffer<float, 1> photonsHashTableCount;
//rtBuffer<Photon, 1> photons;
rtDeclareVariable(uint1, launchIndex, rtLaunchIndex, );

RT_PROGRAM void kernel()
{
    photonsHashTableCount[launchIndex.x] = 0.0f;
}
#endif