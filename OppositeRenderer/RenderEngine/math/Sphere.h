/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "math/Vector3.h"

class Sphere
{
public:
    Vector3 center;
    float radius;
#ifdef __CUDACC__ 
    __device__ 
#endif 
    Sphere()
    {

    }
    Sphere(const Vector3 & center, float radius);
};