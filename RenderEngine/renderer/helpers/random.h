/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once 

#include "renderer/RandomState.h"
#include <stdint.h>

#ifdef USE_CHEAP_RANDOM

/*
The fast/cheap random generation scheme courtesy of 
http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
*/

static uint32_t __host__ __device__ rand_xorshift(uint32_t& state)
{
    state ^= (state << 13);
    state ^= (state >> 17);
    state ^= (state << 5);
    return state;
}

static uint32_t  __host__ __device__ wang_hash(uint32_t seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

static void __host__ __device__ initializeRandomState(RandomState* state, uint32_t seed, uint32_t index)
{
    state[0] = wang_hash(seed+index);
}

// Return a float from [0,1)
static __device__ __inline__ float getRandomUniformFloat( RandomState* state )
{
    float scale = float(0xFFFFFFFF);
    // Clear the last bit to be strictly less than 1
    return float(rand_xorshift(*state) & ~1)/scale;
}

#else

static void __device__ initializeRandomState(RandomState* state, unsigned int seed, unsigned int index)
{
    curand_init(seed+index, 0, 0, state);
}

// Return a float from 0,1
static __device__ __inline__ float getRandomUniformFloat( RandomState* state )
{
    return curand_uniform(state);
}

#endif

static __device__ __inline__ optix::float2 getRandomUniformFloat2( RandomState* state )
{
    optix::float2 sample;
    sample.x = getRandomUniformFloat(state);
    sample.y = getRandomUniformFloat(state);
    return sample;
}