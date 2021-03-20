/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once

#if 0
#define OPTIX_DEBUG_PRINT(depth, str, ...) \
    if(launchIndex.x == 328 && launchIndex.y == 139){ printf("%d %d: ", launchIndex.x, launchIndex.y); \
        for(int i = 0; i < depth; i++){printf(" ");} \
        printf(str, __VA_ARGS__); \
    }
#else
#define OPTIX_DEBUG_PRINT(depth, str, ...) // nothing
#endif

// Create ONB from normalized normal (code: Physically Based Rendering, Pharr & Humphreys pg. 63)

static  __device__ __inline__ void createCoordinateSystem( const optix::float3& N, optix::float3& U, optix::float3& V/*, optix::float3& W*/ )
{
    using namespace optix;

    if(fabs(N.x) > fabs(N.y))
    {
        float invLength = 1.f/sqrtf(N.x*N.x + N.z*N.z);
        U = make_float3(-N.z*invLength, 0.f, N.x*invLength);
    }
    else
    {
        float invLength = 1.f/sqrtf(N.y*N.y + N.z*N.z);
        U = make_float3(0.f, N.z*invLength, -N.y*invLength);
    }
    V = cross(N, U);
}

static __device__ __host__ __inline__ float maxf(float a, float b)
{
    return a > b ? a : b;
}

// Returns true if ray direction points in the opposite direction 
// as the normal, where the normal points outwards from the face
static __device__ __host__ __inline__ bool hitFromOutside(const optix::float3 & rayDirection, const optix::float3 & normal)
{
    return (optix::dot(normal, rayDirection) < 0);
}

static __device__ __inline__ int intmin(int a, int b)
{
    return a < b ? a : b;
}

static __device__ __inline__ float favgf(const optix::float3 & v )
{
    return (v.x+v.y+v.z)*0.3333333333f;
}