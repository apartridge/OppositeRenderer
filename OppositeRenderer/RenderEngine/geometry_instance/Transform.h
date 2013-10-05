/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include <optixu/optixpp_namespace.h>

class Transform
{
private:
    float arr[16];
public:
    Transform();
    optix::Transform getOptixTransform(optix::Context& context);
    void translate( float x, float y, float z );
    void scale( float scale );
};