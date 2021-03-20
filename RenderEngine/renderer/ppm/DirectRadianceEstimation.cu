/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/
#include <cuda.h>
#include <curand_kernel.h>
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "config.h"
#include "renderer/helpers/random.h"
#include "renderer/Light.h"
#include "renderer/RayType.h"
#include "renderer/Hitpoint.h"
#include "renderer/ShadowPRD.h"
#include "renderer/helpers/light.h"

using namespace optix;

rtDeclareVariable(rtObject, sceneRootObject, , );
rtBuffer<Hitpoint, 2> raytracePassOutputBuffer;
rtBuffer<float3, 2> directRadianceBuffer;
rtBuffer<RandomState, 2> randomStates;
rtBuffer<Light, 1> lights;
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(ShadowPRD, shadowPrd, rtPayload, );

RT_PROGRAM void kernel()
{
    Hitpoint rec = raytracePassOutputBuffer[launchIndex];
    
    // Use radiance value if we do not hit a non-specular surface
    if(!(rec.flags & PRD_HIT_NON_SPECULAR))
    {
        if((rec.flags & PRD_HIT_EMITTER) && !(rec.flags & PRD_HIT_SPECULAR))
        {
            directRadianceBuffer[launchIndex] = fminf(rec.radiance, make_float3(1));
        }
        else
        {
            directRadianceBuffer[launchIndex] = rec.radiance;
        }

        return;
    }

    /*
    // Compute direct radiance
    */

    int numLights = lights.size();
    const int numShadowSamples = ENABLE_PARTICIPATING_MEDIA ? 0 : 4;
    float3 directRadiance = make_float3(0);
    if(numShadowSamples > 0)
    {
        float3 avgLightRadiance = make_float3(0.f);

        for(int shadowSample = 0; shadowSample < numShadowSamples; shadowSample++)
        {
            float sample = getRandomUniformFloat(&randomStates[launchIndex]);
            int randomLightIndex = intmin(int(sample*numLights), lights.size()-1);
            Light & light = lights[randomLightIndex];
            float scale = numLights;
            float3 lightContrib = getLightContribution(light, rec.position, rec.normal, sceneRootObject, randomStates[launchIndex]);
            avgLightRadiance += scale * lightContrib;
        }

        directRadiance = rec.attenuation*avgLightRadiance/numShadowSamples;
    }

    directRadianceBuffer[launchIndex] = directRadiance;
    
}

RT_PROGRAM void gatherAnyHitOnNonEmitter()
{
    shadowPrd.attenuation = 0.0f;
    rtTerminateRay();
}