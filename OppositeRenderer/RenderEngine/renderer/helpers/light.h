/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once

#include "renderer/Light.h"
#include "random.h"
#include "helpers.h"
#include "renderer/ShadowPRD.h"
#include "renderer/TransmissionPRD.h"

optix::float3 __inline __device__ getLightContribution(const Light & light, const optix::float3 & rec_position, 
    const optix::float3 & rec_normal, const rtObject & rootObject, RandomState & randomState)
{
    float lightFactor = 1;

    float lightDistance = 0;
    float3 pointOnLight;

    if(light.lightType == Light::AREA)
    {
        float2 sample = getRandomUniformFloat2(&randomState);
        pointOnLight = light.position + sample.x*light.v1 + sample.y*light.v2;
    }
    else if(light.lightType == Light::POINT)
    {
        pointOnLight = light.position;
        lightFactor *= 1.f/4.f;
    }
    else if(light.lightType == Light::SPOT)
    {
        // Todo find correct direct light for spot light
        lightFactor = 0;
    }

    float3 towardsLight = pointOnLight - rec_position;
    lightDistance = optix::length(towardsLight);
    towardsLight = towardsLight / lightDistance;
    float n_dot_l = maxf(0, optix::dot(rec_normal, towardsLight));
    lightFactor *= n_dot_l / (M_PIf*lightDistance*lightDistance);

    if(light.lightType == Light::AREA)
    {
        lightFactor *= maxf(0, optix::dot(-towardsLight, light.normal));
    }

    if (lightFactor > 0.0f)
    {
        ShadowPRD shadowPrd;
        shadowPrd.attenuation = 1.0f;
        optix::Ray shadow_ray (rec_position, towardsLight, RayType::SHADOW, 0.0001, lightDistance-0.0001);
        rtTrace(rootObject, shadow_ray, shadowPrd);
        lightFactor *= shadowPrd.attenuation;
        

        // Check for participating media transmission
/*#if ENABLE_PARTICIPATING_MEDIA
        TransmissionPRD transmissionPrd;
        transmissionPrd.attenuation = 1.0f;
        optix::Ray transmissionRay (rec_position, towardsLight, RayType::PARTICIPATING_MEDIUM_TRANSMISSION, 0.001, lightDistance-0.01);
        rtTrace(rootObject, transmissionRay, transmissionPrd);
        lightFactor *= transmissionPrd.attenuation;
#endif
        */
        //printf("Point on light:%.2f %.2f %.2f shadowPrd.attenuation %.2f\n", pointOnLight.x, pointOnLight.y, pointOnLight.z, shadowPrd.attenuation);
        return light.power*lightFactor;
    }

    return optix::make_float3(0);
};
