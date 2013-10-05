/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#pragma once
#include "Material.h"
class QString;
class Image;
class Texture : public Material
{
public:
    Texture(const QString & textureAbsoluteFilePath);
    Texture(const QString & textureAbsoluteFilePath, const QString & normalMapAbsoluteFilePath);
    virtual ~Texture();
    virtual optix::Material getOptixMaterial(optix::Context & context);
    virtual void registerGeometryInstanceValues(optix::GeometryInstance & instance);

private:
    void loadDiffuseImage( const QString & textureAbsoluteFilePath );
    void loadNormalMapImage( const QString & normalsAbsoluteFilePath );
    optix::TextureSampler createTextureSamplerFromBuffer(optix::Context & context, optix::Buffer buffer);
    optix::Buffer createBufferFromImage(optix::Context & context, const Image & image);

    static bool m_optixMaterialIsCreated;
    static optix::Material m_optixMaterial;
    optix::TextureSampler m_diffuseSampler;
    optix::TextureSampler m_normalMapSampler;
    Image* m_diffuseImage;
    Image* m_normalMapImage;
};