/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "render_engine_export_api.h"
#include <QVector>
#include "RenderServerRenderRequestDetails.h"

class RenderServerRenderRequest
{
public:
    RENDER_ENGINE_EXPORT_API RenderServerRenderRequest();
    RENDER_ENGINE_EXPORT_API RenderServerRenderRequest(unsigned long long sequenceNumber, const QVector<unsigned long long> & iterationNumbers,
                                                      const QVector<double> & ppmRadii, const RenderServerRenderRequestDetails & details);

    RENDER_ENGINE_EXPORT_API ~RenderServerRenderRequest(void);
    RENDER_ENGINE_EXPORT_API const QVector<double> & getPPMRadii() const;
    RENDER_ENGINE_EXPORT_API const QVector<unsigned long long> & getIterationNumbers() const;
    RENDER_ENGINE_EXPORT_API unsigned long long getSequenceNumber() const;
    RENDER_ENGINE_EXPORT_API unsigned long long getFirstIterationNumber() const;
    RENDER_ENGINE_EXPORT_API unsigned int getNumIterations() const;
    RENDER_ENGINE_EXPORT_API const RenderServerRenderRequestDetails & getDetails() const;

private:
    unsigned long long m_sequenceNumber;
    QVector<unsigned long long> m_iterationNumbers;
    QVector<double> m_ppmRadii;
    RenderServerRenderRequestDetails m_details;
};

class QDataStream;
RENDER_ENGINE_EXPORT_API QDataStream & operator << (QDataStream & out, const RenderServerRenderRequest & renderRequest);
RENDER_ENGINE_EXPORT_API QDataStream & operator >> (QDataStream & in, RenderServerRenderRequest & renderRequest);
