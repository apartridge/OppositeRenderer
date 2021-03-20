/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#include "Texture.h"
#include "../renderer/RayType.h"
#include "../util/Image.h"
#include "../util/ptxhelper.h"

#include <QString>

bool Texture::m_optixMaterialIsCreated = false;
optix::Material Texture::m_optixMaterial;

Texture::Texture(const QString & textureAbsoluteFilePath)
    : m_diffuseImage(NULL), m_normalMapImage(NULL)
{
    loadDiffuseImage(textureAbsoluteFilePath);
}

Texture::Texture(const QString & textureAbsoluteFilePath, const QString & normalMapAbosoluteFilePath)
    : m_diffuseImage(NULL), m_normalMapImage(NULL)
{
    loadDiffuseImage(textureAbsoluteFilePath);
    loadNormalMapImage(normalMapAbosoluteFilePath);
}

Texture::~Texture()
{
    delete m_diffuseImage;
    delete m_normalMapImage;
}

void Texture::loadDiffuseImage( const QString & textureAbsoluteFilePath )
{
    try
    {
        m_diffuseImage = new Image(textureAbsoluteFilePath);
    }
    catch(const std::exception & e)
    {
        QString exceptionStr = QString("An error occurred loading of texture: %2").arg(textureAbsoluteFilePath).arg(e.what());
        throw std::runtime_error(exceptionStr.toStdString());
    }
}

void Texture::loadNormalMapImage( const QString & normalMapAbsoluteFilePath )
{
    try
    {
        m_normalMapImage = new Image(normalMapAbsoluteFilePath);
        printf("Loaded normals: %s\n", normalMapAbsoluteFilePath.toStdString().c_str());
    }
    catch(const std::exception & e)
    {
        QString exceptionStr = QString("An error occurred loading of texture's normal map: %1").arg(e.what());
        throw std::runtime_error(exceptionStr.toStdString());
    }
}

optix::Material Texture::getOptixMaterial(optix::Context & context)
{
    if(!m_optixMaterialIsCreated)
    {
        m_optixMaterial = context->createMaterial();
        optix::Program radianceProgram = context->createProgramFromPTXFile(getPtxFile("material/Texture.ptx"), "closestHitRadiance");
        optix::Program photonProgram = context->createProgramFromPTXFile(getPtxFile("material/Texture.ptx"), "closestHitPhoton");

        m_optixMaterial->setClosestHitProgram(RayType::RADIANCE, radianceProgram);
        m_optixMaterial->setClosestHitProgram(RayType::RADIANCE_IN_PARTICIPATING_MEDIUM, radianceProgram);
        m_optixMaterial->setClosestHitProgram(RayType::PHOTON, photonProgram);
        m_optixMaterial->setClosestHitProgram(RayType::PHOTON_IN_PARTICIPATING_MEDIUM, photonProgram);
        m_optixMaterial->validate();
        this->registerMaterialWithShadowProgram(context, m_optixMaterial);
        m_optixMaterialIsCreated = true;
    }

    // Diffuse buffer

    optix::Buffer buffer = createBufferFromImage(context, *m_diffuseImage);
    m_diffuseSampler = createTextureSamplerFromBuffer(context, buffer);

    optix::Buffer normalsBuffer;
    if(m_normalMapImage != NULL)
    {
        normalsBuffer = createBufferFromImage(context, *m_normalMapImage);
    }
    else
    {
        normalsBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, 0, 0);
    }

    m_normalMapSampler = createTextureSamplerFromBuffer(context, normalsBuffer);
    return m_optixMaterial;
}

void Texture::registerGeometryInstanceValues(optix::GeometryInstance & instance )
{
    instance["diffuseSampler"]->setTextureSampler(m_diffuseSampler);
    instance["hasNormals"]->setUint(m_normalMapImage != NULL);
    instance["normalMapSampler"]->setTextureSampler(m_normalMapSampler);
}

optix::TextureSampler Texture::createTextureSamplerFromBuffer( optix::Context & context, optix::Buffer buffer )
{
    optix::TextureSampler sampler = context->createTextureSampler();
    sampler->setWrapMode(0, RT_WRAP_REPEAT);
    sampler->setWrapMode(1, RT_WRAP_REPEAT);
    sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
    sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    sampler->setMaxAnisotropy(4.f);
    sampler->setArraySize(1);
    sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    sampler->setMipLevelCount(1);
    sampler->setBuffer(0, 0, buffer);
    return sampler;
}

optix::Buffer Texture::createBufferFromImage(optix::Context & context, const Image & image )
{
    optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, image.getWidth(), image.getHeight());
    optix::char4* buffer_Host = (optix::char4*)buffer->map();
    memcpy(buffer_Host, image.constData(), image.getWidth()*image.getHeight()*4*sizeof(unsigned char));
    buffer->unmap();
    return buffer;
}