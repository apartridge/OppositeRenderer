/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "RenderServerRenderRequest.h"
#include <QDataStream>

RenderServerRenderRequest::RenderServerRenderRequest(unsigned long long sequenceNumber, const QVector<unsigned long long> & iterationNumbers,
                                                     const QVector<double> & ppmRadii, const RenderServerRenderRequestDetails & details)
    : m_sequenceNumber(sequenceNumber),
      m_iterationNumbers(iterationNumbers), 
      m_ppmRadii(ppmRadii),
      m_details(details)
{

}

RenderServerRenderRequest::RenderServerRenderRequest()
    : m_sequenceNumber(0)
{

}

RenderServerRenderRequest::~RenderServerRenderRequest(void)
{

}

const QVector<unsigned long long> & RenderServerRenderRequest::getIterationNumbers() const
{
    return m_iterationNumbers;
}

const QVector<double> & RenderServerRenderRequest::getPPMRadii() const
{
    return m_ppmRadii;
}

unsigned int RenderServerRenderRequest::getNumIterations() const
{
    return (unsigned int)m_iterationNumbers.size();
}

const RenderServerRenderRequestDetails & RenderServerRenderRequest::getDetails() const
{
    return m_details;
}

unsigned long long RenderServerRenderRequest::getSequenceNumber() const
{
    return m_sequenceNumber;
}

unsigned long long RenderServerRenderRequest::getFirstIterationNumber() const
{
    return m_iterationNumbers.first();
}

QDataStream & operator<<( QDataStream & out, const RenderServerRenderRequest & renderRequest )
{
    QByteArray array;
    QDataStream str(&array, QIODevice::ReadWrite);
    str << (quint64)renderRequest.getSequenceNumber()
        << renderRequest.getIterationNumbers()
        << renderRequest.getPPMRadii() 
        << renderRequest.getDetails();

    out << (int)(array.size()+2*sizeof(int)) << array;
    return out;
}

QDataStream & operator>>( QDataStream & in, RenderServerRenderRequest & renderRequest )
{
    QByteArray array;
    in >> array;
    QDataStream arrayStream(&array, QIODevice::ReadWrite);

    quint64 sequenceNumber;
    QVector<unsigned long long> iterationNumbers;
    QVector<double> ppmRadii;
    RenderServerRenderRequestDetails details;

    arrayStream >> sequenceNumber 
                >> iterationNumbers
                >> ppmRadii 
                >> details;

    renderRequest = RenderServerRenderRequest((unsigned long long)sequenceNumber, iterationNumbers, 
                        ppmRadii, details);

    if(in.status() != QDataStream::Ok)
    {
        printf("Error in RenderServerRenderRequest operator >>.\n");
    }
    return in;
}