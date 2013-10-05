/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

//#define MAX_DEPTH 20

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "config.h"
#include "renderer/Light.h"
#include "renderer/ppm/Photon.h"
#include "renderer/RayType.h"
#include "renderer/Hitpoint.h"
#include "renderer/ppm/PhotonGrid.h"
#include "renderer/RadiancePRD.h"

using namespace optix;

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );

rtBuffer<Photon, 1> photons;
rtBuffer<Hitpoint, 2> raytracePassOutputBuffer;
rtBuffer<float3, 2> indirectRadianceBuffer;

rtDeclareVariable(float, alpha, , );
rtDeclareVariable(float, emittedPhotonsPerIterationFloat, , );
rtDeclareVariable(float, ppmRadius, ,);
rtDeclareVariable(float, ppmRadiusSquared, ,);
rtDeclareVariable(float, ppmRadiusSquaredNew, ,);

#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_UNIFORM_GRID
rtDeclareVariable(uint3, photonsGridSize, , );
rtDeclareVariable(float3, photonsWorldOrigo, ,);
rtDeclareVariable(float, photonsGridCellSize, ,);
rtBuffer<uint, 1> hashmapOffsetTable;
#elif ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_STOCHASTIC_HASH
rtDeclareVariable(uint3, photonsGridSize, , );
rtDeclareVariable(float3, photonsWorldOrigo, ,);
rtDeclareVariable(float, photonsGridCellSize, ,);
rtDeclareVariable(unsigned int, photonsSize, ,);
rtBuffer<unsigned int, 1> photonsHashTableCount;
#elif ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_KD_TREE_CPU
rtBuffer<Photon, 1> photonKdTree;
#endif

#if ENABLE_RENDER_DEBUG_OUTPUT
rtBuffer<uint, 2> debugIndirectRadianceCellsVisisted;
rtBuffer<uint, 2> debugIndirectRadiancePhotonsVisisted;
#endif

__device__ __inline float validPhoton(const Photon & photon, const float distance2, const float radius2, const float3 & hitNormal)
{
    return distance2 <= radius2 && dot(-photon.rayDirection, hitNormal) >= 0; 
}

__device__ __inline float3 photonPower(const Photon & photon, const float distance2, const float radius2)
{
    // Use the gaussian filter from Realistic Image Synthesis Using Photon Mapping, Wann Jensen
    const float alpha = 1.818;
    const float beta = 1.953;
    const float expNegativeBeta = 0.141847;
    float weight = alpha*(1 - (1-exp(-beta*distance2/(2*radius2)))/(1-expNegativeBeta));
    return photon.power*weight;
}

RT_PROGRAM void kernel()
{
    clock_t start = clock();
    Hitpoint rec = raytracePassOutputBuffer[launchIndex];
    
    float3 indirectAccumulatedPower = make_float3( 0.0f, 0.0f, 0.0f );

    int _dPhotonsVisited = 0;
    int _dCellsVisited = 0;

    if(rec.flags & PRD_HIT_NON_SPECULAR)
    {
        float radius2 = ppmRadiusSquared;
        float radius = ppmRadius;

#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_UNIFORM_GRID
        
        float invCellSize = 1.f/photonsGridCellSize;
        float3 normalizedPosition = rec.position - photonsWorldOrigo;
        unsigned int x_lo = (unsigned int)max(0, (int)((normalizedPosition.x - radius) * invCellSize));
        unsigned int y_lo = (unsigned int)max(0, (int)((normalizedPosition.y - radius) * invCellSize));
        unsigned int z_lo = (unsigned int)max(0, (int)((normalizedPosition.z - radius) * invCellSize));
     
        unsigned int x_hi = (unsigned int)min(photonsGridSize.x-1, (unsigned int)((normalizedPosition.x + radius) * invCellSize));
        unsigned int y_hi = (unsigned int)min(photonsGridSize.y-1, (unsigned int)((normalizedPosition.y + radius) * invCellSize));
        unsigned int z_hi = (unsigned int)min(photonsGridSize.z-1, (unsigned int)((normalizedPosition.z + radius) * invCellSize));    

        if(x_lo <= x_hi)
        {
            for(unsigned int z = z_lo; z <= z_hi; z++)
            {
                for(unsigned int y = y_lo; y <= y_hi; y++)
                {
                    optix::uint3 cell;
                    cell.x = x_lo;
                    cell.y = y;
                    cell.z = z;
                    unsigned int from = getPhotonGridIndex1D(cell, photonsGridSize);
                    unsigned int to = from + (x_hi-x_lo);

                    unsigned int offset = hashmapOffsetTable[from];
                    unsigned int offsetTo = hashmapOffsetTable[to+1];
                    unsigned int numPhotons = offsetTo-offset;

                    _dCellsVisited++;

                    for(unsigned int i = offset; i < offset+numPhotons; i++)
                    {
                        const Photon & photon = photons[i];
                        float3 diff = rec.position - photon.position;
                        float distance2 = dot(diff, diff);
                        if(validPhoton(photon, distance2, radius2, rec.normal))
                        {
                            indirectAccumulatedPower += photonPower(photon, distance2, radius2);
                        }
                        _dPhotonsVisited++;
                    }

                }
            }
        }

#elif ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_STOCHASTIC_HASH 
        
        optix::uint3 hitCell = getPhotonGridIndex(rec.position, photonsWorldOrigo, photonsGridCellSize);
     
        #pragma unroll 3
        for(int dz = -1; dz <= 1; dz++)
        {
            #pragma unroll 3
            for(int dy = -1; dy <= 1; dy++)
            {
                #pragma unroll 3
                for(int dx = -1; dx <= 1; dx++)
                {
                    // No hit position can have grid position 0 in x, y or z (because of the padding to the AABB)
                    optix::uint3 cell;
                    cell.x = hitCell.x+dx;
                    cell.y = hitCell.y+dy;
                    cell.z = hitCell.z+dz;
                    _dCellsVisited++;
                    _dPhotonsVisited++;

                    uint hash = getHashValue(cell, photonsGridSize, photonsSize); \
                    const Photon & photon = photons[hash];
                    float3 diff = rec.position - photon.position;
                    float distance2 = dot(diff, diff);
                    if(validPhoton(photon, distance2, radius2, rec.normal))
                    {
                        indirectAccumulatedPower += photonPower(photon, distance2, radius2)*float(photonsHashTableCount[hash]);
                    }
                }
            }
        }
        
#elif ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_KD_TREE_CPU

        // This code is based on the PPM sample in Optix 3.0.0 SDK by NVIDIA

        const size_t MAX_DEPTH = 21;
        unsigned int stack[MAX_DEPTH];
        unsigned int stack_current = 0;
        unsigned int node = 0;
        #define push_node(N) stack[stack_current++] = (N)
        #define pop_node() stack[--stack_current]

        push_node(0);
        do 
        {
            Photon& photon = photonKdTree[ node ];
            _dPhotonsVisited++;
            uint axis = photon.axis;
            if( !( axis & PPM_NULL ) )
            {
                float3 diff = rec.position - photon.position;
                float distance2 = dot(diff, diff);
                if(validPhoton(photon, distance2, radius2, rec.normal))
                {
                    indirectAccumulatedPower += photonPower(photon, distance2, radius2);
                }

                // Recurse
                if( !( axis & PPM_LEAF ) ) {
                    float d;
                    if      ( axis & PPM_X ) d = diff.x;
                    else if ( axis & PPM_Y ) d = diff.y;
                    else                     d = diff.z;
                    // Calculate the next child selector. 0 is left, 1 is right.
                    int selector = d < 0.0f ? 0 : 1;
                    if( d*d < radius2 ) {
                        push_node( (node<<1) + 2 - selector );
                    }
                    node = (node<<1) + 1 + selector;
                } else {
                    node = pop_node();
                }
            } else {
                node = pop_node();
            }
        }
        while ( node );
#endif
    }

    float3 indirectRadiance = indirectAccumulatedPower * rec.attenuation * (1.0f/(M_PIf*ppmRadiusSquared)) *  (1.0f/emittedPhotonsPerIterationFloat);

    // Add contribution from volumetric radiance
#if ENABLE_PARTICIPATING_MEDIA
    indirectRadiance += rec.volumetricRadiance / emittedPhotonsPerIterationFloat;
#endif

    indirectRadianceBuffer[launchIndex] = indirectRadiance;

#if ENABLE_RENDER_DEBUG_OUTPUT
    debugIndirectRadianceCellsVisisted[launchIndex] = _dCellsVisited;
    debugIndirectRadiancePhotonsVisisted[launchIndex] = _dPhotonsVisited;
#endif

}