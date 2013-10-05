/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "config.h"
#include "renderer/RadiancePRD.h"
#include "renderer/HitPoint.h"
#include "renderer/RayType.h"
#include "renderer/Camera.h"
#include "renderer/helpers/random.h"
#include "renderer/helpers/camera.h"

using namespace optix;

rtDeclareVariable(rtObject, sceneRootObject, , );
rtBuffer<Hitpoint, 2> raytracePassOutputBuffer;
rtBuffer<RandomState, 2> randomStates;
rtDeclareVariable(float, ppmDefaultRadius2, , );
rtDeclareVariable(Camera, camera, , );
rtDeclareVariable(float, camera_aperture, , );
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(float, iterationNumber, , );
rtDeclareVariable(RadiancePRD, radiancePrd, rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void generateRay()
{
    RadiancePRD radiancePrd;
    radiancePrd.attenuation = make_float3( 1.0f );
    radiancePrd.radiance = make_float3(0.f);
    radiancePrd.depth = 0;
    radiancePrd.flags = 0;
    radiancePrd.randomState = randomStates[launchIndex];
#if ENABLE_PARTICIPATING_MEDIA
    radiancePrd.volumetricRadiance = make_float3(0);
#endif

    float2 screen = make_float2(raytracePassOutputBuffer.size());
    float2 sample = getRandomUniformFloat2(&radiancePrd.randomState);
    float2 d = ( make_float2(launchIndex) + sample ) / screen * 2.0f - 1.0f;
    float3 rayOrigin = camera.eye;
    float3 rayDirection = normalize(d.x*camera.camera_u + d.y*camera.camera_v + camera.lookdir);

    modifyRayForDepthOfField(camera, rayOrigin, rayDirection, radiancePrd.randomState);
    
    Ray ray(rayOrigin, rayDirection, RayType::RADIANCE, 0.001f);
    rtTrace(sceneRootObject, ray, radiancePrd);
    
    // Store as a PPM Hitpoint
    Hitpoint & rec = raytracePassOutputBuffer[launchIndex];
    rec.position = radiancePrd.position; 
    rec.normal = radiancePrd.normal;
    rec.attenuation = radiancePrd.attenuation;
    rec.radiance = radiancePrd.radiance;
    rec.flags = radiancePrd.flags;
#if ENABLE_PARTICIPATING_MEDIA
    rec.volumetricRadiance = radiancePrd.volumetricRadiance;
#endif
    randomStates[launchIndex] = radiancePrd.randomState;
}

//
// Miss program
//

RT_PROGRAM void miss()
{
    radiancePrd.flags = PRD_MISS;
    radiancePrd.attenuation = make_float3(0.f);
    radiancePrd.radiance = 0*make_float3(5.f);
}

//
// Exception handler program
//

RT_PROGRAM void exception()
{
    printf("Exception Radiance!\n");
    radiancePrd.flags = PRD_ERROR;
    radiancePrd.attenuation = make_float3(0.f,0.f,1.f);
}