/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "RenderServerRenderRequestDetails.h"
#include <QDataStream>

RenderServerRenderRequestDetails::RenderServerRenderRequestDetails()
{

}

RenderServerRenderRequestDetails::RenderServerRenderRequestDetails( const Camera & camera, QByteArray sceneName, RenderMethod::E renderMethod, 
                                                                    unsigned int width, unsigned int height, double ppmAlpha ) :
  m_camera(camera), m_sceneName(sceneName), m_renderMethod(renderMethod), m_width(width), m_height(height), m_ppmAlpha(ppmAlpha)
{

}

double RenderServerRenderRequestDetails::getPPMAlpha() const
{
    return m_ppmAlpha;
}

const Camera & RenderServerRenderRequestDetails::getCamera() const
{
    return m_camera;
}

const RenderMethod::E RenderServerRenderRequestDetails::getRenderMethod() const
{
    return m_renderMethod;
}

unsigned int RenderServerRenderRequestDetails::getWidth() const
{
    return m_width;
}

unsigned int RenderServerRenderRequestDetails::getHeight() const
{
    return m_height;
}


const QByteArray & RenderServerRenderRequestDetails::getSceneName() const
{
    return m_sceneName;
}

QDataStream & operator<<( QDataStream & out, const RenderServerRenderRequestDetails & details )
{
    QByteArray array;
    QDataStream str(&array, QIODevice::ReadWrite);
    str << details.getCamera() 
        << details.getSceneName()
        << (quint32)details.getRenderMethod()
        << (quint32)details.getWidth() 
        << (quint32)details.getHeight()
        << (double)details.getPPMAlpha();

    out << array;
    return out;
}

QDataStream & operator>>( QDataStream & in, RenderServerRenderRequestDetails & details )
{
    QByteArray array;
    in >> array;
    QDataStream arrayStream(&array, QIODevice::ReadWrite);

    Camera camera;
    QByteArray sceneName;
    quint32 renderMethod;
    quint32 width, height;
    double ppmAlpha;

    arrayStream 
        >> camera 
        >> sceneName 
        >> renderMethod 
        >> width 
        >> height
        >> ppmAlpha;

    details = RenderServerRenderRequestDetails(camera, sceneName, (RenderMethod::E)renderMethod, width, height, ppmAlpha);

    if(in.status() != QDataStream::Ok)
    {
        printf("Error in RenderServerRenderRequestDetails operator >>.\n");
    }

    return in;
}
