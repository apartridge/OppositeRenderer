/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#include "DiffuseEmitter.h"
#include "renderer/RayType.h"

bool DiffuseEmitter::m_optixMaterialIsCreated = false;
optix::Material DiffuseEmitter::m_optixMaterial;

DiffuseEmitter::DiffuseEmitter(const Vector3 & power, const Vector3 & Kd)
    : m_power(power), m_Kd(Kd), m_inverseArea(0)
{
    m_power.x *= Kd.x;
    m_power.y *= Kd.y;
    m_power.z *= Kd.z;
}

optix::Material DiffuseEmitter::getOptixMaterial(optix::Context & context)
{
    if(!m_optixMaterialIsCreated)
    {
        optix::Program radianceProgram = context->createProgramFromPTXFile( "DiffuseEmitter.cu.ptx", "closestHitRadiance");
        m_optixMaterial = context->createMaterial();
        m_optixMaterial->setClosestHitProgram(RayType::RADIANCE, radianceProgram);
        m_optixMaterial->setClosestHitProgram(RayType::RADIANCE_IN_PARTICIPATING_MEDIUM, radianceProgram);
        m_optixMaterial->setClosestHitProgram(RayType::PHOTON, context->createProgramFromPTXFile( "DiffuseEmitter.cu.ptx", "closestHitPhoton") );
        m_optixMaterial->setAnyHitProgram(RayType::SHADOW, context->createProgramFromPTXFile( "DiffuseEmitter.cu.ptx", "gatherAnyHitOnEmitter"));
        this->registerMaterialWithShadowProgram(context, m_optixMaterial);
        m_optixMaterialIsCreated = true;
    }
    return m_optixMaterial;
}

/*
// Register any material-dependent values to be available in the optix program.
*/

void DiffuseEmitter::registerGeometryInstanceValues(optix::GeometryInstance & instance )
{
    instance["Kd"]->setFloat(m_Kd);
    instance["Ks"]->setFloat(0.0f, 0.0f, 0.0f);
    optix::float3 powerPerArea = m_power;

    powerPerArea.x *= m_inverseArea;
    powerPerArea.y *= m_inverseArea;
    powerPerArea.z *= m_inverseArea;

    instance["powerPerArea"]->setFloat(powerPerArea);
    instance["power"]->setFloat(m_power);
}

Vector3 DiffuseEmitter::getPower() const
{
    return m_power;
}

void DiffuseEmitter::setInverseArea( float inverseArea )
{
    m_inverseArea = inverseArea;
}
