/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "Vector3.h"
#include <math.h>

Vector3 Vector3::operator+( const Vector3 & other ) const
{
    return Vector3(this->x + other.x, this->y + other.y, this->z + other.z);
}

Vector3 Vector3::operator-( const Vector3 & other ) const
{
    return Vector3(this->x - other.x, this->y - other.y, this->z - other.z);
}

Vector3 Vector3::operator*( float scale ) const
{
    return Vector3(this->x*scale, this->y*scale, this->z*scale);
}

float Vector3::length2()
{
    return dot(*this, *this);
}

float Vector3::dot( const Vector3 & a, const Vector3 & b )
{
    return a.x*b.x + a.y*b.y + a.z*b.x;
}

float Vector3::length()
{
    return sqrtf(length2());
}