/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once

#include "GeometryInstance.h"
#include "material/Material.h"
#include "math/Sphere.h"

class Sphere;
class SphereInstance : public GeometryInstance
{
private:
    Sphere m_sphere;
    static optix::Program m_programBoundingBox;
    static optix::Program m_programIntersection;
    static bool m_hasLoadedOptixPrograms;
public:
    SphereInstance(Material& material, const Sphere & sphere);
    virtual optix::Geometry getOptixGeometry(optix::Context & context);
};