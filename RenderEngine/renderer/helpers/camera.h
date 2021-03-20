/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "samplers.h"

// If the camera has a aperture > 0, we model it using a thin lens model. Camera.lookdir length is the focal length of the camera.
__inline__ __device__ void modifyRayForDepthOfField(const Camera & camera, optix::float3 & rayOrigin, optix::float3& rayDirection, RandomState& randomState )
{
    if(camera.aperture > 0)
    {
        optix::float3 focalPlaneCenterPoint = camera.eye + camera.lookdir;
        optix::float3 camLookDir = optix::normalize(camera.lookdir);

        float focalPlaneT = (optix::dot(camLookDir, focalPlaneCenterPoint) - optix::dot(camLookDir, camera.eye))/optix::dot(camLookDir, rayDirection); 
        optix::float3 lookAt = rayOrigin + focalPlaneT*rayDirection;

        optix::float2 dofSample = getRandomUniformFloat2(&randomState);
        float2 disc = sampleUnitDisc(dofSample);

        rayOrigin += disc.x*camera.camera_u*camera.aperture + disc.y*camera.camera_v*camera.aperture;
        rayDirection = optix::normalize(lookAt - rayOrigin);
    }
}