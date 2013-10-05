/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "RenderResultPacket.h"
#include <QDataStream>
#include <QVector>

RenderResultPacket::RenderResultPacket()
{

}

RenderResultPacket::RenderResultPacket(unsigned long long sequenceNumber, QVector<unsigned long long> iterationNumbersInPacket, 
    QByteArray output)
    : 
    m_sequenceNumber(sequenceNumber),
    m_iterationNumbersInPacket(iterationNumbersInPacket),
    m_output(output),
    m_renderTimeSeconds(0),
    m_totalTimeSeconds(0)
{

}

RenderResultPacket::~RenderResultPacket( void )
{

}

float RenderResultPacket::getRenderTimeSeconds() const
{
    return m_renderTimeSeconds;
}

float RenderResultPacket::getTotalTimeSeconds() const
{
    return m_totalTimeSeconds;
}

void RenderResultPacket::setRenderTimeSeconds( float renderTime )
{
    m_renderTimeSeconds = renderTime;
}

void RenderResultPacket::setTotalTimeSeconds( float totalTime )
{
    m_totalTimeSeconds = totalTime;
}

unsigned long long RenderResultPacket::getSequenceNumber() const
{
    return m_sequenceNumber;
}

const QByteArray & RenderResultPacket::getOutput() const
{
    return m_output;
}

// Return a list of iteration numbers in packet which is sorted
const QVector<unsigned long long> & RenderResultPacket::getIterationNumbersInPacket() const
{
    return m_iterationNumbersInPacket;
}

int RenderResultPacket::getNumIterationsInPacket() const
{
    return m_iterationNumbersInPacket.size();
}

unsigned long long RenderResultPacket::getFirstIterationNumber() const
{
    return m_iterationNumbersInPacket.first();
}

bool RenderResultPacket::operator<( const RenderResultPacket & other ) const
{
    return this->getFirstIterationNumber() < other.getFirstIterationNumber();
}

unsigned long long RenderResultPacket::getLastIterationNumber() const
{
    return m_iterationNumbersInPacket.last();
}

// Merge other into this render result packet

void RenderResultPacket::merge( const RenderResultPacket & other )
{
    int thisIterations = this->getIterationNumbersInPacket().size();
    int otherIterations = other.getIterationNumbersInPacket().size();
    int numPixels = this->getOutput().size()/sizeof(float);
    const float* inputData = (const float*)other.getOutput().constData();
    float* outputData = (float*)this->getOutput().data();
    float scale = 1.f/(thisIterations+otherIterations);
    for(unsigned int i = 0; i < numPixels; i += 3)
    {
        outputData[i]   = (thisIterations*outputData[i]   + otherIterations*inputData[i]  ) * scale;
        outputData[i+1] = (thisIterations*outputData[i+1] + otherIterations*inputData[i+1]) * scale;
        outputData[i+2] = (thisIterations*outputData[i+2] + otherIterations*inputData[i+2]) * scale;
    }
    m_iterationNumbersInPacket += other.getIterationNumbersInPacket();
}

QDataStream & operator<<( QDataStream & out, const RenderResultPacket & results )
{
    QByteArray output = results.getOutput();
    QVector<unsigned long long> iterationNumbersInPacket = results.getIterationNumbersInPacket();
    qSort(iterationNumbersInPacket);

    // Send size of packet as the first 64 bits so that receiver knows how much data to expect
    // The size of the different values are listed in http://qt-project.org/doc/qt-4.8/datastreamformat.html

    quint64 sizeOutputBuffer = (quint64)(output.size() + sizeof(quint32));
    quint64 sizeIterationNumbersInPacketVector = (quint64)(iterationNumbersInPacket.size()*sizeof(unsigned long long) + sizeof(quint32));

    quint64 size = sizeOutputBuffer + sizeIterationNumbersInPacketVector;
    size += (quint64)sizeof(quint64) + 2*(quint64)sizeof(float);

    out << size 
        << (quint64)results.getSequenceNumber()
        << iterationNumbersInPacket
        << results.getRenderTimeSeconds() 
        << results.getTotalTimeSeconds()
        << output;
    return out;
}

QDataStream & operator>>( QDataStream & in, RenderResultPacket & results )
{
    unsigned long long sequenceNumber;
    QVector<unsigned long long> iterationNumbersInPacket;
    QByteArray output;
    float renderTimeSeconds;
    float totalTimeSeconds;

    in >> (quint64)sequenceNumber;
    in >> iterationNumbersInPacket;
    in >> renderTimeSeconds;
    in >> totalTimeSeconds;
    in >> output;
    
    results = RenderResultPacket(sequenceNumber, iterationNumbersInPacket, output);
    results.setRenderTimeSeconds(renderTimeSeconds);
    results.setTotalTimeSeconds(totalTimeSeconds);
    return in;
}
