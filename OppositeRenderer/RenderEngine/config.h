/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once

#define PPM_X         ( 1 << 0 )
#define PPM_Y         ( 1 << 1 )
#define PPM_Z         ( 1 << 2 )
#define PPM_LEAF      ( 1 << 3 )
#define PPM_NULL      ( 1 << 4 )

#define ACCELERATION_STRUCTURE_UNIFORM_GRID 0
#define ACCELERATION_STRUCTURE_KD_TREE_CPU 1
#define ACCELERATION_STRUCTURE_STOCHASTIC_HASH 2
#define ACCELERATION_STRUCTURE (ACCELERATION_STRUCTURE_UNIFORM_GRID)

#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_STOCHASTIC_HASH
#define MAX_PHOTONS_DEPOSITS_PER_EMITTED 1
#else
#define MAX_PHOTONS_DEPOSITS_PER_EMITTED 4
#endif

#define ENABLE_RENDER_DEBUG_OUTPUT 0
#define ENABLE_PARTICIPATING_MEDIA 0

#define MAX_PHOTON_TRACE_DEPTH (ENABLE_PARTICIPATING_MEDIA?15:7)
#define MAX_RADIANCE_TRACE_DEPTH 9
#define NUM_VOLUMETRIC_PHOTONS 200000
#define PHOTON_TRACING_RR_START_DEPTH 3
#define PATH_TRACING_RR_START_DEPTH 3