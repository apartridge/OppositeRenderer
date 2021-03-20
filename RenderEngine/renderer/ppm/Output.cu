/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtBuffer<float3, 2> outputBuffer;
rtBuffer<float3, 2> indirectRadianceBuffer;
rtBuffer<float3, 2> directRadianceBuffer;
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex, );
rtDeclareVariable(uint, localIterationNumber, , );

static __device__ __inline float3 averageInNewRadiance(const float3 newRadiance, const float3 oldRadiance, const unsigned int iterationNumber)
{
    // If iterationNumber = 0, we do not average but take new
    if(iterationNumber > 0)
    {
        return oldRadiance + (newRadiance-oldRadiance)/float(iterationNumber+1);
    }
    else
    {
        return newRadiance;
    }
}

RT_PROGRAM void kernel()
{
    float3 finalRadiance = directRadianceBuffer[launchIndex] + indirectRadianceBuffer[launchIndex];
    outputBuffer[launchIndex] = averageInNewRadiance(finalRadiance, outputBuffer[launchIndex], localIterationNumber);
}