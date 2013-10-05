/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#include "config.h"


#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "renderer/Hitpoint.h"
#include "renderer/helpers/random.h"
#include "renderer/helpers/helpers.h"
#include "renderer/helpers/samplers.h"
#include "renderer/RayType.h"
#include "renderer/RadiancePRD.h"
#include "renderer/ShadowPRD.h"
#include "renderer/ppm/VolumetricRadiancePRD.h"
#include "renderer/ppm/PhotonPRD.h"
#include "renderer/TransmissionPRD.h"
#include "renderer/ppm/Photon.h"
#include "renderer/ppm/PhotonGrid.h"

using namespace optix;

rtDeclareVariable(rtObject, volumetricPhotonsRoot, , );
rtDeclareVariable(rtObject, sceneRootObject, , );
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(RadiancePRD, radiancePrd, rtPayload, );
rtDeclareVariable(VolumetricRadiancePRD, volRadiancePrd, rtPayload, );
rtDeclareVariable(ShadowPRD, shadowPrd, rtPayload, );
rtDeclareVariable(PhotonPRD, photonPrd, rtPayload, );
rtDeclareVariable(TransmissionPRD, transmissionPrd, rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, tHit, rtIntersectionDistance, );

rtDeclareVariable(float3, geometricNormal, attribute geometricNormal, ); 
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, ); 

#if ENABLE_PARTICIPATING_MEDIA
rtBuffer<Photon, 1> photons;
rtBuffer<Photon, 1> volumetricPhotons;
rtDeclareVariable(float, sigma_a, , );
rtDeclareVariable(float, sigma_s, , );
rtDeclareVariable(float3, Ks, , );
rtDeclareVariable(float3, Kd, , );
rtDeclareVariable(float, indexOfRefraction, , );
rtDeclareVariable(uint, maxPhotonDepositsPerEmitted, , );
#endif

RT_PROGRAM void closestHitRadiance()
{
#if ENABLE_PARTICIPATING_MEDIA
    const float sigma_t = sigma_a + sigma_s;
    float3 worldShadingNormal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shadingNormal));
    float3 hitPoint = ray.origin + tHit*ray.direction;
    bool isHitFromOutside = hitFromOutside(ray.direction, worldShadingNormal);
    double tHitStack = tHit + 0.1 - 0.1; // Important, prevents compiler optimization on variable

    /*OPTIX_DEBUG_PRINT(0, "Hit media (%.2f %.2f %.2f) %s (attn: %.2f %.2f  %.2f)\n", hitPoint.x, hitPoint.y, hitPoint.z, isHitFromOutside ? "outside" : "inside", 
        radiancePrd.attenuation.x, radiancePrd.attenuation.y, radiancePrd.attenuation.z);*/
    
    if(isHitFromOutside)
    {
        float3 attenSaved = radiancePrd.attenuation + 0.1 - 0.1; // Important, prevents compiler optimization on variable

        // Send ray through the medium
        Ray newRay(hitPoint, ray.direction, RayType::RADIANCE_IN_PARTICIPATING_MEDIUM, 0.01);
        rtTrace(sceneRootObject, newRay, radiancePrd);

        float distance = radiancePrd.lastTHit;
        float transmittance = exp(-distance*sigma_t);

        VolumetricRadiancePRD volRadiancePrd;
        volRadiancePrd.radiance = make_float3(0);
        volRadiancePrd.numHits = 0;
        volRadiancePrd.sigma_t = sigma_t;
        volRadiancePrd.sigma_s = sigma_s;

        // Get volumetric radiance

        Ray ray(hitPoint, ray.direction, RayType::VOLUMETRIC_RADIANCE, 0.0000001, distance);
        rtTrace(volumetricPhotonsRoot, ray, volRadiancePrd);
        
        // Multiply existing volumetric transmittance with current transmittance, and add gathered volumetric radiance
        // from this path

        radiancePrd.volumetricRadiance *= transmittance;
        radiancePrd.volumetricRadiance += attenSaved*volRadiancePrd.radiance;
        radiancePrd.attenuation *= transmittance;
    }
    else
    {
        // We are escaping the boundary of the participating medium, so we'll compute the attenuation and volumetric radiance for the remaining path
        // and deliver it to a parent stack frame.
        Ray newRay = Ray(hitPoint, ray.direction, RayType::RADIANCE, 0.01);
        rtTrace(sceneRootObject, newRay, radiancePrd);
    }

    radiancePrd.lastTHit = tHitStack;
#endif
}

/*
// 
*/

RT_PROGRAM void closestHitPhoton()
{
#if ENABLE_PARTICIPATING_MEDIA
    const float sigma_t = sigma_a + sigma_s;
    
    photonPrd.depth++;

    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    //float3 worldGeometricNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometricNormal ) );
    //float3 ffnormal = faceforward( worldShadingNormal, ray.direction, worldGeometricNormal );
    float3 hitPoint = ray.origin + tHit*ray.direction;
    bool hitInside = (dot(worldShadingNormal, ray.direction) > 0);


    // If we hit from the inside with a PHOTON_IN_PARTICIPATING_MEDIUM ray, we have escaped the boundry of the medium.
    // We move the ray just a tad to the outside and continue ray tracing there

    if(hitInside && ray.ray_type == RayType::PHOTON_IN_PARTICIPATING_MEDIUM)
    {
        //OPTIX_DEBUG_PRINT(photonPrd.depth-1, "Hit medium P(%.2f %.2f %.2f) from inside: move past\n", hitPoint.x, hitPoint.y, hitPoint.z);
        Ray newRay = Ray(hitPoint+0.0001*ray.direction, ray.direction, RayType::PHOTON, 0.001, RT_DEFAULT_MAX);
        rtTrace(sceneRootObject, newRay, photonPrd);
        return;
    }

    //OPTIX_DEBUG_PRINT(photonPrd.depth-1, "Hit medium %s P(%.2f %.2f %.2f) RT=%d\n", hitInside ? "inside" : "outside", hitPoint.x, hitPoint.y, hitPoint.z, ray.ray_type);

    float sample = getRandomUniformFloat(&photonPrd.randomState);
    float scatterLocationT = - logf(1-sample)/sigma_t;
    float3 scatterPosition = hitPoint + scatterLocationT*ray.direction;
    int depth = photonPrd.depth;

    if(launchIndex.x == 185000)
    printf("", photonPrd.depth-1);// line necessary due to optix bug...

    //OPTIX_DEBUG_PRINT(photonPrd.depth-1, "Probing [0,t] ...\n");

    // We need to see if anything obstructs the ray in the interval from the hitpoint to the scatter location.
    // If nothings obstructs then we scatter at eventPosition. Otherwise, the photon continues on its path and we don't do anything
    // when we return to this stack frame. We keep the photonPRD depth on the stack to compare it when the rtTrace returns.

    Ray newRay(hitPoint, ray.direction, RayType::PHOTON_IN_PARTICIPATING_MEDIUM, 0.001, scatterLocationT);
    rtTrace(sceneRootObject, newRay, photonPrd);
#ifdef OPTIX_MATERIAL_DUMP
    for(int i = 0; i<depth;i++) printf("\t");
#endif
    // If depth is unmodified, no surface was hit from hitpoint to scatterLocation, so we store it as a scatter event.
    // We also scatter a photon in a new direction sampled by the phase function at this location.

    if(depth == photonPrd.depth)
    {
        const float scatterAlbedo = sigma_s/sigma_t;

        if(getRandomUniformFloat(&photonPrd.randomState) >= scatterAlbedo)
        {
            return;
        }

        //photonPrd.power *= scatterAlbedo;

        // Store photon at scatter location
        //if(photonPrd.numStoredPhotons < maxPhotonDepositsPerEmitted)
        {
            int volumetricPhotonIdx = photonPrd.pm_index % NUM_VOLUMETRIC_PHOTONS;
            volumetricPhotons[volumetricPhotonIdx].power = photonPrd.power;
            volumetricPhotons[volumetricPhotonIdx].position = scatterPosition;
            atomicAdd(&volumetricPhotons[volumetricPhotonIdx].numDeposits, 1);
        }
        
        // Check if we have gone above max number of photons or stack depth
        if(photonPrd.depth >= MAX_PHOTON_TRACE_DEPTH)
        {
            return;
        }

        // Create the scattered ray with a direction given by importance sampling of the phase function

        float3 scatterDirection = sampleUnitSphere(getRandomUniformFloat2(&photonPrd.randomState));

        OPTIX_DEBUG_PRINT(photonPrd.depth-1, "Not interrupted. Store, scatter P(%.2f %.2f %.2f) D(%.2f %.2f %.2f)\n", scatterPosition.x, scatterPosition.y, scatterPosition.z, 
            scatterDirection.x, scatterDirection.y, scatterDirection.z);

        Ray scatteredRay(scatterPosition, scatterDirection, RayType::PHOTON, 0.001, RT_DEFAULT_MAX);
        rtTrace(sceneRootObject, scatteredRay, photonPrd);
        
    }
    else
    {
        //OPTIX_DEBUG_PRINT(depth-1, "Found surface in [0,t], no scatter!\n");
    }
#endif
}
