/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#pragma once
#include "Material.h"
#include "math/Vector3.h"

class Diffuse : public Material
{
private:
    Vector3 Kd;
    static bool m_optixMaterialIsCreated;
    static optix::Material m_optixMaterial;
public:
    Diffuse(const Vector3 & Kd);
    virtual optix::Material getOptixMaterial(optix::Context & context);
    virtual void registerGeometryInstanceValues(optix::GeometryInstance & instance);
};