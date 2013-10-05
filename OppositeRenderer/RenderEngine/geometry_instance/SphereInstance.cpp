/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "SphereInstance.h"
#include "math/Sphere.h"

bool SphereInstance::m_hasLoadedOptixPrograms = false;
optix::Program SphereInstance::m_programBoundingBox;
optix::Program SphereInstance::m_programIntersection;

SphereInstance::SphereInstance(Material & _material, const Sphere & sphere) 
    : GeometryInstance(_material),
      m_sphere(sphere)
{

}

optix::Geometry SphereInstance::getOptixGeometry( optix::Context & context)
{
    if(m_hasLoadedOptixPrograms == false)
    {
        m_programBoundingBox = context->createProgramFromPTXFile("Sphere.cu.ptx", "boundingBox");
        m_programIntersection = context->createProgramFromPTXFile("Sphere.cu.ptx", "intersect");
        m_hasLoadedOptixPrograms = true;
    }

    optix::Geometry sphere = context->createGeometry();
    sphere->setPrimitiveCount(1);
    sphere->setIntersectionProgram( m_programIntersection );
    sphere->setBoundingBoxProgram( m_programBoundingBox );

    sphere["radius"]->setFloat( this->m_sphere.radius );
    sphere["center"]->setFloat( this->m_sphere.center );

    return sphere;
}