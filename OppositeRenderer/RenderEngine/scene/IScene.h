/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once

#include <optixu/optixpp_namespace.h>
#include <QVector>
#include "renderer/Camera.h"
#include "renderer/Light.h"
#include "render_engine_export_api.h"
#include "math/AAB.h"

class IScene
{
public:
    RENDER_ENGINE_EXPORT_API IScene();
    RENDER_ENGINE_EXPORT_API virtual ~IScene();
    RENDER_ENGINE_EXPORT_API virtual optix::Group getSceneRootGroup(optix::Context & context) = 0;
    RENDER_ENGINE_EXPORT_API virtual const QVector<Light> & getSceneLights() const = 0;
    RENDER_ENGINE_EXPORT_API virtual Camera getDefaultCamera() const = 0;
    RENDER_ENGINE_EXPORT_API virtual const char* getSceneName() const = 0;
    RENDER_ENGINE_EXPORT_API virtual AAB getSceneAABB() const = 0;
    RENDER_ENGINE_EXPORT_API virtual float getSceneInitialPPMRadiusEstimate() const;
    RENDER_ENGINE_EXPORT_API virtual unsigned int getNumTriangles() const = 0;
};