/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "config.h"
#include "renderer/RayType.h"
#include "renderer/RadiancePRD.h"
#include "renderer/ppm/PhotonPRD.h"
#include "renderer/ppm/Photon.h"
#include "renderer/helpers/random.h"
#include "renderer/helpers/samplers.h"
#include "renderer/helpers/store_photon.h"

using namespace optix;

rtDeclareVariable(rtObject, sceneRootObject, , );
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(RadiancePRD, radiancePrd, rtPayload, );
rtDeclareVariable(PhotonPRD, photonPrd, rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, tHit, rtIntersectionDistance, );

rtDeclareVariable(float3, geometricNormal, attribute geometricNormal, ); 
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, ); 
rtDeclareVariable(float3, tangent, attribute tangent, ); 
rtDeclareVariable(float3, bitangent, attribute bitangent, ); 
rtDeclareVariable(float2, textureCoordinate, attribute textureCoordinate, ); 

rtBuffer<Photon, 1> photons;
rtTextureSampler<uchar4, 2, cudaReadModeNormalizedFloat> diffuseSampler;
rtTextureSampler<uchar4, 2, cudaReadModeNormalizedFloat> normalMapSampler;
rtDeclareVariable(unsigned int, hasNormals, , );
rtDeclareVariable(uint, maxPhotonDepositsPerEmitted, , );

#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_STOCHASTIC_HASH
rtDeclareVariable(uint3, photonsGridSize, , );
rtDeclareVariable(float3, photonsWorldOrigo, ,);
rtDeclareVariable(float, photonsGridCellSize, ,);
rtDeclareVariable(unsigned int, photonsSize,,);
rtBuffer<unsigned int, 1> photonsHashTableCount;
#endif


/*
// Radiance Program
*/

__inline__ __device__ float3 getNormalMappedNormal(const float3 & normal, const float3 & tangent, const float3 & bitangent, const float4 & normalMap)
{
    float4 nMap = 2*normalMap - 1;
    float3 N;
    N.x = nMap.x*tangent.x + nMap.y*bitangent.x + nMap.z*normal.x;
    N.y = nMap.x*tangent.y + nMap.y*bitangent.y + nMap.z*normal.y;
    N.z = nMap.x*tangent.z + nMap.y*bitangent.z + nMap.z*normal.z;
    return normalize(N);
}

RT_PROGRAM void closestHitRadiance()
{
    float3 worldShadingNormal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shadingNormal));
    float3 hitPoint = ray.origin + tHit*ray.direction;

    float3 normal = worldShadingNormal;
    if(hasNormals)
    {
        float3 worldTangent = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, tangent));
        float3 worldBitangent = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, bitangent));
        normal = getNormalMappedNormal(worldShadingNormal, worldTangent, worldBitangent, 
                        tex2D(normalMapSampler, textureCoordinate.x, textureCoordinate.y));
    }

    radiancePrd.flags |= PRD_HIT_NON_SPECULAR;
    radiancePrd.normal = normal;
    radiancePrd.position = hitPoint;
    radiancePrd.lastTHit = tHit;
    
    if(radiancePrd.flags & PRD_PATH_TRACING)
    {
        radiancePrd.randomNewDirection = sampleUnitHemisphereCos(worldShadingNormal, getRandomUniformFloat2(&radiancePrd.randomState));
    }

    float4 value = tex2D( diffuseSampler, textureCoordinate.x, textureCoordinate.y );
    float3 value3 = make_float3(value.x, value.y, value.z);
    radiancePrd.attenuation *= value3;
}

/*
// Photon Program
*/

RT_PROGRAM void closestHitPhoton()
{
    float3 worldShadingNormal = normalize(rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal));
    float3 normal = worldShadingNormal;
    if(hasNormals)
    {
        float3 worldTangent = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, tangent));
        float3 worldBitangent = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, bitangent));
        normal = getNormalMappedNormal(worldShadingNormal, worldTangent, worldBitangent, 
                        tex2D(normalMapSampler, textureCoordinate.x, textureCoordinate.y));
    }
    float3 hitPoint = ray.origin + tHit*ray.direction;
    float3 newPhotonDirection;

    // Record hit if it has bounced at least once
    if(photonPrd.depth >= 1)
    {
        Photon photon (photonPrd.power, hitPoint, ray.direction, worldShadingNormal);
        STORE_PHOTON(photon);
    }

    float4 value = tex2D(diffuseSampler, textureCoordinate.x, textureCoordinate.y);
    float3 value3 = make_float3(value.x, value.y, value.z);
    photonPrd.power *= value3;
#ifdef OPTIX_MATERIAL_DUMP
    for(int i = 0; i<photonPrd.depth;i++) printf("\t"); 
        printf("Hit diffuse at P(%.2f %.2f %.2f) t=%.3f\n", hitPoint.x, hitPoint.y, hitPoint.z, tHit);
#endif

    photonPrd.weight *= fmaxf(value3);

    // Use russian roulette sampling from depth X to limit the length of the path

    if( photonPrd.depth >= PHOTON_TRACING_RR_START_DEPTH)
    {
        float probContinue = favgf(value3);
        float probSample = getRandomUniformFloat(&photonPrd.randomState);
        if(probSample >= probContinue )
        {
            return;
        }
        photonPrd.power /= probContinue;
    }

    photonPrd.depth++;
    if(photonPrd.depth >= MAX_PHOTON_TRACE_DEPTH || photonPrd.weight < 0.01)
    {
        return;
    }

#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_UNIFORM_GRID || ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_KD_TREE_CPU
    if(photonPrd.numStoredPhotons >= maxPhotonDepositsPerEmitted)
        return;
#endif

    newPhotonDirection = sampleUnitHemisphereCos(worldShadingNormal, getRandomUniformFloat2(&photonPrd.randomState));
    optix::Ray newRay( hitPoint, newPhotonDirection, RayType::PHOTON, 0.01 );
    rtTrace(sceneRootObject, newRay, photonPrd);
}
