/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "helpers.h"

// Get a random direction from the hemisphere of direction around normalized normal, 
// sampled with the cosine distribution p(theta, phi) = cos(theta)/PI

static __device__ __inline__ optix::float3 sampleUnitHemisphereCos(const optix::float3 & normal, const optix::float2& sample)
{
    using namespace optix;

    float theta = acosf(sqrtf(sample.x));
    float phi = 2.0f * M_PIf *sample.y;
    float xs = sinf(theta) * cosf(phi);
    float ys = cosf(theta);
    float zs = sinf(theta) * sinf(phi);

    float3 U, V;
    createCoordinateSystem(normal, U, V);

    return optix::normalize(xs*U + ys*normal + zs*V);
}

// Sample unit hemisphere around (normalized) normal

static __device__ __inline__ optix::float3 sampleUnitHemisphere(const optix::float3 & normal, const optix::float2& sample)
{
    optix::float3 U, V;
    createCoordinateSystem( normal, U, V);
    float phi = 2.0f * M_PIf*sample.x;
    float r = sqrtf( sample.y );
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = 1.0f - x*x -y*y;
    z = z > 0.0f ? sqrtf(z) : 0.0f;
    return optix::normalize(U*x + V*y + normal*z);
}

static __device__ __inline__ optix::float3 sampleUnitSphere(const optix::float2& sample)
{
    optix::float3 v;
    v.z = sample.x;
    float t =  2*M_PIf*sample.y;
    float r = sqrtf(1.f-v.z*v.z);
    v.x = r*cosf(t);
    v.y = r*sinf(t);
    return v;
}

static __device__ __inline__ optix::float2 sampleUnitDisc(const optix::float2& sample)
{
    float r = sqrtf(sample.x);
    float theta = 2.f*M_PIf*sample.y;
    float x = r*cosf(theta);
    float y = r*sinf(theta);
    return make_float2(x, y);
}

// Sample disc (normal must be normalized)

static __device__ float3 sampleDisc(const float2 & sample, const float3 & center, const float radius, const float3 & normal)
{
    float3 U, V;
    createCoordinateSystem( normal, U, V);
    float2 unitDisc = sampleUnitDisc(sample);
    return center + radius * ( U*unitDisc.x + V*unitDisc.y );
}

