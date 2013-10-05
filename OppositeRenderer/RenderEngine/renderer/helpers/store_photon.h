/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "renderer/ppm/PhotonGrid.h"

// Unfortunately, we need a macro for photon storing code

#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_UNIFORM_GRID || ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_KD_TREE_CPU
#define STORE_PHOTON(photon) \
    photons[photonPrd.pm_index + photonPrd.numStoredPhotons] = photon; \
    photonPrd.numStoredPhotons++;
#else
#define STORE_PHOTON(photon) \
    { \
    uint3 gridLoc = getPhotonGridIndex(photon.position, photonsWorldOrigo, photonsGridCellSize); \
    uint hash = getHashValue(gridLoc, photonsGridSize, photonsSize); \
    photons[hash] = photon; \
    atomicAdd(&photonsHashTableCount[hash], 1); \
    }
#endif