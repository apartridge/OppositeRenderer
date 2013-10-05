/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "AABInstance.h"

bool AABInstance::m_hasLoadedOptixPrograms = false;
optix::Program AABInstance::m_programBoundingBox;
optix::Program AABInstance::m_programIntersection;

AABInstance::AABInstance(Material & _material, const AAB & aab) 
    : GeometryInstance(_material),
      m_aab(aab)
{

}

optix::Geometry AABInstance::getOptixGeometry( optix::Context & context)
{
    if(m_hasLoadedOptixPrograms == false)
    {
        m_programBoundingBox = context->createProgramFromPTXFile("AAB.cu.ptx", "boundingBox");
        m_programIntersection = context->createProgramFromPTXFile("AAB.cu.ptx", "intersect");
        m_hasLoadedOptixPrograms = true;
    }
    
    optix::Geometry geometry = context->createGeometry();
    geometry->setPrimitiveCount(1u);
    geometry->setBoundingBoxProgram(m_programBoundingBox);
    geometry->setIntersectionProgram(m_programIntersection);

    geometry["cuboidMin"]->setFloat( this->m_aab.min );
    geometry["cuboidMax"]->setFloat( this->m_aab.max );

    return geometry;
}