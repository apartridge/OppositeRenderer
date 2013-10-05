/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "config.h"

static __host__ __device__ __inline optix::uint3 floor3f(const optix::float3 & f)
{
    optix::uint3 result;
    result.x = (unsigned int)floor(f.x);
    result.y = (unsigned int)floor(f.y);
    result.z = (unsigned int)floor(f.z);
    return result;
}

__host__ __device__ __inline optix::uint3 getPhotonGridIndex(const optix::float3 & position, const optix::float3 & sceneWorldOrigo, const float & cellSize)
{
    float3 positivePosition = position-sceneWorldOrigo;
    return floor3f(positivePosition*(1.f/cellSize));
}

__host__ __device__ __inline unsigned int getPhotonGridIndex1D(const optix::uint3 & gridPosition, const optix::uint3& gridSize )
{
    return gridPosition.x + gridPosition.y*gridSize.x + gridPosition.z*gridSize.x*gridSize.y;
}

__host__ __device__ __inline unsigned int getHashValue(const optix::uint3 & gridPosition, const optix::uint3& gridSize, const unsigned int max)
{
    return getPhotonGridIndex1D(gridPosition, gridSize) & (max-1);
}
