/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#ifndef CORNELL_H
#define CORNELL_H

#include "Scene.h"
#include "renderer/Light.h"
#include "renderer/Camera.h"
#include "render_engine_export_api.h"


class Material;

class Cornell : public IScene
{
public:
    RENDER_ENGINE_EXPORT_API Cornell(void);
    RENDER_ENGINE_EXPORT_API virtual ~Cornell(void){}
    RENDER_ENGINE_EXPORT_API virtual optix::Group getSceneRootGroup(optix::Context & context);
    RENDER_ENGINE_EXPORT_API virtual const QVector<Light> & getSceneLights() const;
    RENDER_ENGINE_EXPORT_API virtual Camera getDefaultCamera(void) const;
    RENDER_ENGINE_EXPORT_API virtual const char* getSceneName() const;
    RENDER_ENGINE_EXPORT_API static const char* getCornellSceneName();
    RENDER_ENGINE_EXPORT_API virtual unsigned int getNumTriangles() const;
    RENDER_ENGINE_EXPORT_API virtual AAB getSceneAABB() const;

private:
    optix::Material m_material;
    optix::Material m_glassMaterial;
    optix::Program m_pgram_bounding_box;
    optix::Program m_pgram_intersection;
    QVector<Light> m_sceneLights;
    AAB m_sceneAABB;
    optix::GeometryInstance createParallelogram(optix::Context & context, const optix::float3& anchor, const optix::float3& offset1, const optix::float3& offset2, Material & material);

};
#endif