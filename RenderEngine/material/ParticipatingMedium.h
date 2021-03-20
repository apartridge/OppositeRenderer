/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#pragma once
#include "Material.h"

class ParticipatingMedium : public Material
{
public:
    ParticipatingMedium(float sigma_s, float sigma_a);
    virtual optix::Material getOptixMaterial(optix::Context & context);
    virtual void registerGeometryInstanceValues(optix::GeometryInstance & instance);
private:
    //float indexOfRefraction;
    static bool m_optixMaterialIsCreated;
    static optix::Material m_optixMaterial;
    float m_sigma_s;
    float m_sigma_a;
};