/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "AAB.h"

AAB::AAB(const Vector3 & min, const Vector3 & max)
{
    this->min = min;
    this->max = max;
}

AAB::AAB()
    : min(Vector3()), max(Vector3())
{

}

Vector3 AAB::getExtent() const
{
    return this->max - this->min;
}

Vector3 AAB::getCenter() const
{
    return (min+max)*0.5f;
}

Sphere AAB::getBoundingSphere() const
{
    Vector3 center = getCenter();
    Vector3 extent = getExtent();
    float radius = (max - center).length();
    Sphere sphere(center, radius);
    return sphere;
}
