/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "Transform.h"

Transform::Transform()
{
    this->arr[0] = 1;
    this->arr[1] = 0;
    this->arr[2] = 0;
    this->arr[3] = 0;
    this->arr[4] = 0;
    this->arr[5] = 1;
    this->arr[6] = 0;
    this->arr[7] = 0;
    this->arr[8] = 0;
    this->arr[9] = 0;
    this->arr[10] = 1;
    this->arr[11] = 0;
    this->arr[12] = 0;
    this->arr[13] = 0;
    this->arr[14] = 0;
    this->arr[15] = 1;
}

optix::Transform Transform::getOptixTransform( optix::Context& context )
{
    optix::Transform t = context->createTransform();
    t->setMatrix(false, this->arr, NULL);
    return t;
}

void Transform::translate( float x, float y, float z )
{
    this->arr[3] += x;
    this->arr[7] += y;
    this->arr[11] += z;
}

void Transform::scale( float scale )
{
    this->arr[0] *= scale;
    this->arr[5] *= scale;
    this->arr[10] *= scale;
}