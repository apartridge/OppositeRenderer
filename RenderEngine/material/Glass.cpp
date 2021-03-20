/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#include "Glass.h"
#include "../renderer/RayType.h"
#include "../util/ptxhelper.h"

bool Glass::m_optixMaterialIsCreated = false;
optix::Material Glass::m_optixMaterial;

Glass::Glass(float indexOfRefraction, const Vector3& Ks)
{
    this->indexOfRefraction = indexOfRefraction;
    this->Ks = Ks;
}

optix::Material Glass::getOptixMaterial(optix::Context& context)
{
    if (!m_optixMaterialIsCreated)
    {
        m_optixMaterial = context->createMaterial();
        optix::Program radianceClosestProgram
            = context->createProgramFromPTXFile(getPtxFile("material/Glass.ptx"), "closestHitRadiance");
        optix::Program radianceAnyHitProgram
            = context->createProgramFromPTXFile(getPtxFile("material/Glass.ptx"), "anyHitRadiance");
        optix::Program photonClosestProgram
            = context->createProgramFromPTXFile(getPtxFile("material/Glass.ptx"), "closestHitPhoton");
        optix::Program photonAnyHitProgram
            = context->createProgramFromPTXFile(getPtxFile("material/Glass.ptx"), "anyHitPhoton");

        m_optixMaterial->setClosestHitProgram(static_cast<int>(RayType::RADIANCE), radianceClosestProgram);
        // m_optixMaterial->setAnyHitProgram(RayType::RADIANCE, radianceAnyHitProgram );
        m_optixMaterial->setClosestHitProgram(static_cast<int>(RayType::RADIANCE_IN_PARTICIPATING_MEDIUM), radianceClosestProgram);
        // m_optixMaterial->setAnyHitProgram(RayType::RADIANCE_IN_PARTICIPATING_MEDIUM, radianceAnyHitProgram);

        m_optixMaterial->setClosestHitProgram(static_cast<int>(RayType::PHOTON), photonClosestProgram);
        // m_optixMaterial->setAnyHitProgram(RayType::PHOTON, photonAnyHitProgram);
        m_optixMaterial->setClosestHitProgram(static_cast<int>(RayType::PHOTON_IN_PARTICIPATING_MEDIUM), photonClosestProgram);
        // m_optixMaterial->setAnyHitProgram(RayType::PHOTON_IN_PARTICIPATING_MEDIUM, photonAnyHitProgram);

        this->registerMaterialWithShadowProgram(context, m_optixMaterial);
        m_optixMaterialIsCreated = true;
    }

    return m_optixMaterial;
}

/*
// Register any material-dependent values to be available in the optix program.
*/
void Glass::registerGeometryInstanceValues(optix::GeometryInstance& instance)
{
    instance["indexOfRefraction"]->setFloat(this->indexOfRefraction);
    instance["Kd"]->setFloat(0, 0, 0);
    instance["Ks"]->setFloat(this->Ks);
}
