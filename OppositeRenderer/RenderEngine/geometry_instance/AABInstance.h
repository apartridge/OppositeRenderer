/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once

#include <optixu/optixpp_namespace.h>
#include "GeometryInstance.h"
#include "material/Material.h"
#include "math/AAB.h"

class AABInstance : public GeometryInstance
{
public:
    AABInstance(Material& material, const AAB & aab);
private:
    AAB m_aab;
    virtual optix::Geometry getOptixGeometry(optix::Context & context);

    static optix::Program m_programBoundingBox;
    static optix::Program m_programIntersection;
    static bool m_hasLoadedOptixPrograms;

};