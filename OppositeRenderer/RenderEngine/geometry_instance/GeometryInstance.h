/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include <optixu/optixpp_namespace.h>

class Material;
class GeometryInstance
{
protected:
    Material & m_material;
    GeometryInstance(Material & _material) : m_material(_material) {};
protected:
    virtual optix::Geometry getOptixGeometry( optix::Context & context) = 0;
public:
    optix::GeometryInstance getOptixGeometryInstance(optix::Context & context);
};