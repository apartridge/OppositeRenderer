/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once

#include "render_engine_export_api.h"
#include "renderer/Camera.h"
#include "renderer/RenderMethod.h"
#include <QString>
#include <QByteArray>

class RenderServerRenderRequestDetails
{
public:
    RENDER_ENGINE_EXPORT_API RenderServerRenderRequestDetails();
    RENDER_ENGINE_EXPORT_API RenderServerRenderRequestDetails(const Camera & camera, QByteArray sceneName, RenderMethod::E renderMethod, unsigned int width, unsigned int height, double ppmAlpha);
    RENDER_ENGINE_EXPORT_API unsigned int getWidth() const;
    RENDER_ENGINE_EXPORT_API unsigned int getHeight() const;
    RENDER_ENGINE_EXPORT_API double getPPMAlpha() const;
    RENDER_ENGINE_EXPORT_API const Camera & getCamera() const;
    RENDER_ENGINE_EXPORT_API const QByteArray & getSceneName() const;
    RENDER_ENGINE_EXPORT_API const RenderMethod::E getRenderMethod() const;
private:
    Camera m_camera;
    RenderMethod::E m_renderMethod;
    unsigned int m_width;
    unsigned int m_height;
    double m_ppmAlpha;
    QByteArray m_sceneName;
};

class QDataStream;
RENDER_ENGINE_EXPORT_API QDataStream & operator << (QDataStream & out, const RenderServerRenderRequestDetails & details);
RENDER_ENGINE_EXPORT_API QDataStream & operator >> (QDataStream & in, RenderServerRenderRequestDetails & details);
