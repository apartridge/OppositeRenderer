/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "config.h"
#include "renderer/Hitpoint.h"
#include "renderer/helpers/random.h"
#include "renderer/helpers/helpers.h"
#include "renderer/RayType.h"
#include "renderer/RadiancePRD.h"
#include "renderer/ppm/PhotonPRD.h"

using namespace optix;

//
// Scene wide variables
//

rtDeclareVariable(rtObject, sceneRootObject, , );

//
// Ray generation program
//

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );

//
// Closest hit material
//

rtDeclareVariable(float3, Ks, , );
rtDeclareVariable(float3, Kd, , );
rtDeclareVariable(float, indexOfRefraction, , );
rtDeclareVariable(float3, geometricNormal, attribute geometricNormal, ); 
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, ); 
rtDeclareVariable(RadiancePRD, radiancePrd, rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, tHit, rtIntersectionDistance, );

struct IndexOfRefractions
{
    float n1;
    float n2;
};

__device__ __inline bool willTravelInsideGlass (bool hitFromOutside, bool reflection)
{
    return hitFromOutside && !reflection || !hitFromOutside && reflection;
}

__device__ __inline IndexOfRefractions getIndexOfRefractions(bool hitFromOutside, float glassIOR)
{
    IndexOfRefractions i;
    
    if(hitFromOutside)
    {
        i.n1 = 1;
        i.n2 = glassIOR;
    }
    else
    {
        i.n1 = glassIOR;
        i.n2 = 1;
    }
    return i;
}

__device__ __inline float reflectionFactor(float cosI, float cosT, float n1, float n2)
{
    float rp = (n2*cosI - n1*cosT)/(n2*cosI+n1*cosT);
    float rs = (n1*cosI - n2*cosT)/(n1*cosI+n2*cosT);
    return ( rp*rp + rs*rs ) / 2.f ;
}

RT_PROGRAM void closestHitRadiance()
{
    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    bool isHitFromOutside = hitFromOutside(ray.direction, worldShadingNormal);
    float3 N = isHitFromOutside ? worldShadingNormal : -worldShadingNormal;
    float3 hitPoint = ray.origin + tHit*ray.direction;

    //OPTIX_DEBUG_PRINT("Radiance hit glass %s P(%.2f %.2f %.2f)\n", isHitFromOutside ? "outside" : "inside", hitPoint.x, hitPoint.y, hitPoint.z);

    IndexOfRefractions ior = getIndexOfRefractions(isHitFromOutside, indexOfRefraction);
    float3 refractionDirection;
    bool validRefraction = refract(refractionDirection, ray.direction, N, ior.n2/ior.n1);
    float cosThetaI = -dot(ray.direction, N);
    float cosThetaT = -dot(refractionDirection, N);

    // Find reflection factor using Fresnel equation
    float reflFactor = validRefraction ? reflectionFactor(cosThetaI, cosThetaT, ior.n1, ior.n2) : 1.f;

    float sample = getRandomUniformFloat(&radiancePrd.randomState);
    bool isReflected = (sample <= reflFactor);
    float3 newRayDirection;

    if(isReflected)
    {
        newRayDirection = reflect(ray.direction, N);
    }
    else
    {
        newRayDirection = refractionDirection;
        radiancePrd.attenuation *= (ior.n2*ior.n2)/(ior.n1*ior.n1);
    }

    radiancePrd.flags |= PRD_HIT_SPECULAR;
    radiancePrd.flags &= ~PRD_HIT_NON_SPECULAR;
    
    radiancePrd.depth++;
    // If we will travel inside the glass object, we set type to be RayType::RADIANCE_IN_PARTICIPATING_MEDIUM to avoid intersecting the 
    // participating media

    bool travellingInside = willTravelInsideGlass(isHitFromOutside, isReflected);
    RayType::E rayType = travellingInside ? RayType::RADIANCE_IN_PARTICIPATING_MEDIUM : RayType::RADIANCE;
    Ray newRay = Ray(hitPoint, newRayDirection, rayType, 0.0001, RT_DEFAULT_MAX );

    if(radiancePrd.depth <= MAX_RADIANCE_TRACE_DEPTH)
    {
        rtTrace( sceneRootObject, newRay, radiancePrd );
    }
    else
    {
        radiancePrd.attenuation *= 0;
    }

    radiancePrd.lastTHit = tHit;
}


RT_PROGRAM void anyHitRadiance()
{
    float3 worldShadingNormal = rtTransformNormal(RT_OBJECT_TO_WORLD, shadingNormal);
    bool isHitFromOutside = hitFromOutside(ray.direction, worldShadingNormal);
    if( (isHitFromOutside && ray.ray_type == RayType::RADIANCE_IN_PARTICIPATING_MEDIUM) || 
        (!isHitFromOutside && ray.ray_type == RayType::RADIANCE ))
    {
        //printf("Ignore int' tHit=%.4f", tHit);
        rtIgnoreIntersection();
    }
}

/*
// Pass the photon along its way through the glass
*/

rtDeclareVariable(PhotonPRD, photonPrd, rtPayload, );

RT_PROGRAM void closestHitPhoton()
{
    float3 worldShadingNormal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shadingNormal ) );
    bool isHitFromOutside = hitFromOutside(ray.direction, worldShadingNormal);
    float3 N = isHitFromOutside ? worldShadingNormal : -worldShadingNormal;
    float3 hitPoint = ray.origin + tHit*ray.direction;
   
    
    IndexOfRefractions ior = getIndexOfRefractions(isHitFromOutside, indexOfRefraction);
    float3 refractionDirection;
    bool validRefraction = refract(refractionDirection, ray.direction, N, ior.n2/ior.n1);
    float cosThetaI = -dot(ray.direction, N);
    float cosThetaT = -dot(refractionDirection, N);

    // Find reflection factor using Fresnel equation
    float reflFactor = validRefraction ? reflectionFactor(cosThetaI, cosThetaT, ior.n1, ior.n2) : 1.f;
    float sample = getRandomUniformFloat(&photonPrd.randomState);
    bool isReflected = (sample <= reflFactor);
    float3 newRayDirection;

    if(isReflected)
    {
        newRayDirection = reflect(ray.direction, N);
    }
    else
    {
        newRayDirection = refractionDirection;
    }

    OPTIX_DEBUG_PRINT(photonPrd.depth, "Photon hit glass %s (%s) %s P(%.2f %.2f %.2f)\n", isHitFromOutside ? "outside" : "inside",
        willTravelInsideGlass(isHitFromOutside, isReflected)  ? "will travel inside" : "will travel outside", isReflected ? "reflect":"refract", hitPoint.x, hitPoint.y, hitPoint.z);

    photonPrd.depth++;
    if (photonPrd.depth <= MAX_PHOTON_TRACE_DEPTH)
    {
        // If we are going to travel inside the glass object, set ray type to RayType::PHOTON_IN_PARTICIPATING_MEDIUM to
        // prevent interaction with any partcipating medium
        RayType::E rayType = willTravelInsideGlass(isHitFromOutside, isReflected) ? RayType::PHOTON_IN_PARTICIPATING_MEDIUM : RayType::PHOTON;
        Ray newRay(hitPoint, newRayDirection, rayType, 0.0001);
        rtTrace(sceneRootObject, newRay, photonPrd);
    }
}

RT_PROGRAM void anyHitPhoton()
{
    float3 worldShadingNormal = rtTransformNormal(RT_OBJECT_TO_WORLD, shadingNormal);
    bool isHitFromOutside = hitFromOutside(ray.direction, worldShadingNormal);
    if( (isHitFromOutside && ray.ray_type == RayType::PHOTON_IN_PARTICIPATING_MEDIUM) || 
        (!isHitFromOutside && ray.ray_type == RayType::PHOTON ))
    {
        //printf("Ignore int' tHit=%.4f", tHit);
        rtIgnoreIntersection();
    }
}