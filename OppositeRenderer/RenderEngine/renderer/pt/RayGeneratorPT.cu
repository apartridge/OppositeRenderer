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
#include "renderer/ShadowPRD.h"
#include "renderer/Light.h"
#include "renderer/Camera.h"
#include "renderer/helpers/random.h"
#include "renderer/helpers/light.h"
#include "renderer/helpers/camera.h"

using namespace optix;

rtDeclareVariable(rtObject, sceneRootObject, , );
rtDeclareVariable(Camera, camera, , );
rtBuffer<Light, 1> lights;
rtBuffer<float3, 2> outputBuffer;
rtBuffer<RandomState, 2> randomStates;
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint, localIterationNumber, , );
rtDeclareVariable(RadiancePRD, radiancePrd, rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(int, ptDirectLightSampling, ,);

static __device__ __inline float3 averageInNewRadiance(const float3 newRadiance, const float3 oldRadiance, const float localIterationNumber)
{
    if(localIterationNumber >= 1)
    {
        return oldRadiance + (newRadiance-oldRadiance)/(localIterationNumber+1);
    }
    else
    {
        return newRadiance;
    }
}

RT_PROGRAM void generateRay()
{
    RadiancePRD radiancePrd;
    radiancePrd.attenuation = make_float3( 1.0f );
    radiancePrd.radiance = make_float3(0.f);
    radiancePrd.depth = 0u; 
    radiancePrd.randomState = randomStates[launchIndex];

    float2 screen = make_float2( outputBuffer.size() );
    float2 sample = getRandomUniformFloat2(&radiancePrd.randomState);
    float2 d = ( make_float2(launchIndex) + sample ) / screen * 2.0f - 1.0f;

    float3 rayOrigin = camera.eye;
    float3 rayDirection = normalize(d.x*camera.camera_u + d.y*camera.camera_v + camera.lookdir);

    modifyRayForDepthOfField(camera, rayOrigin, rayDirection, radiancePrd.randomState);

    Ray ray = Ray(rayOrigin, rayDirection, RayType::RADIANCE, 0.001);

    float3 finalRadiance = make_float3(0);

    int numShadowSamples = ptDirectLightSampling ? 1 : 0;
    int numPaths = ptDirectLightSampling ? 5 : 10;

    for(int i = 0; i < numPaths; i++) // Paths
    {
        radiancePrd.flags = PRD_PATH_TRACING;
        rtTrace(sceneRootObject, ray, radiancePrd);

        if(radiancePrd.flags & PRD_HIT_EMITTER)
        {
            if(radiancePrd.flags & PRD_HIT_SPECULAR || i == 0)
            {
                finalRadiance = radiancePrd.radiance;
            }
            break;
        }
        else if(radiancePrd.flags & PRD_HIT_NON_SPECULAR)
        {
            int numLights = lights.size();
            float3 accumLightRadiancePreBrdf = make_float3(0);

            for(int shadowSample = 0; shadowSample < numShadowSamples; shadowSample++)
            {
                int randomLightIndex = int(getRandomUniformFloat(&randomStates[launchIndex])*numLights);
                Light & light = lights[randomLightIndex];
                float scale = numLights;

                float3 lightContrib = scale*getLightContribution(light, radiancePrd.position, radiancePrd.normal, sceneRootObject, radiancePrd.randomState);
                accumLightRadiancePreBrdf += lightContrib;
            }

            float3 directRadiance = radiancePrd.attenuation*accumLightRadiancePreBrdf/numShadowSamples;
            finalRadiance += directRadiance;
            
            ray.origin = radiancePrd.position;
            ray.direction = radiancePrd.randomNewDirection;
        }
        else
        {
            break;
        }

        if(i >= PATH_TRACING_RR_START_DEPTH) // Russian Roulette sampling
        {
            float sample = getRandomUniformFloat(&randomStates[launchIndex]);
            float probabilityContinue = fmaxf(radiancePrd.attenuation);
            if(sample > probabilityContinue)
            {
                break;
            }
            radiancePrd.attenuation /= probabilityContinue;
        }
    }
 
    // Write outputbuffer radiance value
    outputBuffer[launchIndex] = averageInNewRadiance(finalRadiance, outputBuffer[launchIndex], localIterationNumber);
    randomStates[launchIndex] = radiancePrd.randomState;
}

//
// Exception handler program
//

RT_PROGRAM void exception()
{
    printf("Exception Radiance PT!\n");
    radiancePrd.flags = PRD_ERROR;
    radiancePrd.attenuation = make_float3(0,0,1);
}

//
// Miss program
//

RT_PROGRAM void miss()
{
    radiancePrd.flags = PRD_MISS;
    radiancePrd.attenuation = make_float3(0,0,0);
}
