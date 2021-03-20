/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#pragma once
#include "../math/Vector3.h"
#include "Material.h"

class Mirror : public Material
{
public:
    Mirror(const Vector3& Kr);
    virtual optix::Material getOptixMaterial(optix::Context& context);
    virtual void registerGeometryInstanceValues(optix::GeometryInstance& instance);

private:
    Vector3 Kr;
    static bool m_optixMaterialIsCreated;
    static optix::Material m_optixMaterial;
};