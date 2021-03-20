/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once

namespace OptixEntryPoint
{
    enum  {
        PPM_RAYTRACE_PASS,
        PPM_PHOTON_PASS,
        PPM_INDIRECT_RADIANCE_ESTIMATION_PASS,
        PPM_DIRECT_RADIANCE_ESTIMATION_PASS,
        PPM_OUTPUT_PASS,
        PT_RAYTRACE_PASS,
#if ENABLE_PARTICIPATING_MEDIA
        PPM_CLEAR_VOLUMETRIC_PHOTONS_PASS,
#endif
#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_STOCHASTIC_HASH
        PPM_CLEAR_PHOTONS_UNIFORM_GRID_PASS,
#endif
        NUM_PASSES
    };
}