/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "Photon.h"

#if ENABLE_PARTICIPATING_MEDIA

using namespace optix;

rtBuffer<Photon, 1> volumetricPhotons;
rtDeclareVariable(uint1, launchIndex, rtLaunchIndex, );

RT_PROGRAM void kernel()
{
    Photon photon = Photon(make_float3(0), make_float3(0), make_float3(0), make_float3(0));
    photon.numDeposits = 0;
    volumetricPhotons[launchIndex.x] = photon;
}

#endif