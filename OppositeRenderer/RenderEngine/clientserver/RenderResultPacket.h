/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "render_engine_export_api.h"
#include <QByteArray>
#include <QVector>

/*
A RenderResultPacket is what we send from server to client with the rendered image.
A packet can consist of several iterations of the algorithm combined in a single image/frame to save space.
There is a vector of iteration numbers in each packet which says what this packet contains.
*/

class RenderResultPacket
{
public:
    RENDER_ENGINE_EXPORT_API RenderResultPacket();
    RENDER_ENGINE_EXPORT_API RenderResultPacket(unsigned long long sequenceNumber, QVector<unsigned long long> iterationNumbersInPacket, 
        QByteArray output);

    RENDER_ENGINE_EXPORT_API ~RenderResultPacket(void);
    RENDER_ENGINE_EXPORT_API unsigned long long getSequenceNumber() const;
    RENDER_ENGINE_EXPORT_API const QVector<unsigned long long> & getIterationNumbersInPacket() const;
    RENDER_ENGINE_EXPORT_API int getNumIterationsInPacket() const;
    RENDER_ENGINE_EXPORT_API float getRenderTimeSeconds() const;
    RENDER_ENGINE_EXPORT_API float getTotalTimeSeconds() const;
    RENDER_ENGINE_EXPORT_API void setRenderTimeSeconds(float renderTime);
    RENDER_ENGINE_EXPORT_API void setTotalTimeSeconds(float renderTime);
    RENDER_ENGINE_EXPORT_API unsigned long long getFirstIterationNumber() const;
    RENDER_ENGINE_EXPORT_API unsigned long long getLastIterationNumber() const;
    RENDER_ENGINE_EXPORT_API const QByteArray & getOutput() const;
    RENDER_ENGINE_EXPORT_API void merge(const RenderResultPacket & other);
    RENDER_ENGINE_EXPORT_API bool operator < (const RenderResultPacket & other) const;

private:
    unsigned long long m_sequenceNumber;
    QVector<unsigned long long> m_iterationNumbersInPacket;
    QByteArray m_output;
    float m_renderTimeSeconds;
    float m_totalTimeSeconds;
};

class QDataStream;
RENDER_ENGINE_EXPORT_API QDataStream & operator << (QDataStream & out, const RenderResultPacket & renderRequest);
RENDER_ENGINE_EXPORT_API QDataStream & operator >> (QDataStream & in, RenderResultPacket & renderFrameCommand);
