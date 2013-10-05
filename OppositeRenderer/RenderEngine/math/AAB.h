/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "Vector3.h"
#include "math/Sphere.h"

class AAB
{
public:
    Vector3 min;
    Vector3 max;
    AAB(const Vector3 &, const Vector3 &);
    AAB();
    Vector3 getExtent() const;
    Vector3 getCenter() const;
    Sphere getBoundingSphere() const;
    void addPadding(float a)
    {
        min -= a;
        max += a;
    }
};