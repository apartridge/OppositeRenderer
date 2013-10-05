/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#pragma once
#include "Material.h"
#include "math/Vector3.h"

class DiffuseEmitter : public Material
{
private:
    Vector3 m_power;
    Vector3 m_Kd;
    static bool m_optixMaterialIsCreated;
    static optix::Material m_optixMaterial;
    float m_inverseArea;
public:
    DiffuseEmitter(const Vector3 & power, const Vector3 & Kd);
    virtual optix::Material getOptixMaterial(optix::Context & context);
    virtual void registerGeometryInstanceValues(optix::GeometryInstance & instance);
    Vector3 getPower() const;
    void setInverseArea(float inverseArea);
};