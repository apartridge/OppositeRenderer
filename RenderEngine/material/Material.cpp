/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#include "Material.h"
#include "../renderer/RayType.h"
#include "../util/ptxhelper.h"

bool Material::m_hasLoadedOptixAnyHitProgram = false;
optix::Program Material::m_optixAnyHitProgram;

Material::~Material()
{
}

void Material::registerMaterialWithShadowProgram(optix::Context& context, optix::Material& material)
{
    if (!m_hasLoadedOptixAnyHitProgram)
    {
        m_optixAnyHitProgram = context->createProgramFromPTXFile(
            getPtxFile("renderer/ppm/DirectRadianceEstimation.ptx"), "gatherAnyHitOnNonEmitter");
        m_hasLoadedOptixAnyHitProgram = true;
    }
    material->setAnyHitProgram(static_cast<int>(RayType::SHADOW), m_optixAnyHitProgram);
}