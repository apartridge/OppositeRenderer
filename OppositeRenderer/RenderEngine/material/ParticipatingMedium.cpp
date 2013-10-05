/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#include "ParticipatingMedium.h"
#include "renderer/RayType.h"

bool ParticipatingMedium::m_optixMaterialIsCreated = false;
optix::Material ParticipatingMedium::m_optixMaterial;

ParticipatingMedium::ParticipatingMedium(float sigma_s, float sigma_a)
    : m_sigma_a(sigma_a), m_sigma_s(sigma_s)
{
}

optix::Material ParticipatingMedium::getOptixMaterial(optix::Context & context)
{
    if(!m_optixMaterialIsCreated)
    {
        optix::Program radianceProgram = context->createProgramFromPTXFile( "ParticipatingMedium.cu.ptx", "closestHitRadiance");
        optix::Program photonProgram = context->createProgramFromPTXFile( "ParticipatingMedium.cu.ptx", "closestHitPhoton");
        //optix::Program transmissionProgram = context->createProgramFromPTXFile( "ParticipatingMedium.cu.ptx", "radianceTransmission");

        m_optixMaterial = context->createMaterial();
        m_optixMaterial->setClosestHitProgram(RayType::RADIANCE, radianceProgram);
        m_optixMaterial->setClosestHitProgram(RayType::RADIANCE_IN_PARTICIPATING_MEDIUM, radianceProgram);
        m_optixMaterial->setClosestHitProgram(RayType::PHOTON, photonProgram);
        m_optixMaterial->setClosestHitProgram(RayType::PHOTON_IN_PARTICIPATING_MEDIUM, photonProgram);
        
        this->registerMaterialWithShadowProgram(context, m_optixMaterial);

        m_optixMaterialIsCreated = true;
    }

    return m_optixMaterial;
}

/*
// Register any material-dependent values to be available in the optix program.
*/

void ParticipatingMedium::registerGeometryInstanceValues(optix::GeometryInstance & instance )
{
    instance["participatingMedium"]->setUint(1);
    instance["sigma_a"]->setFloat(m_sigma_a);
    instance["sigma_s"]->setFloat(m_sigma_s);

}
