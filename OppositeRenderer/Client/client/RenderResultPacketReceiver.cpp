/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "RenderResultPacketReceiver.hxx"
#include "RenderServerConnection.hxx"
#include "DistributedApplication.hxx"
#include "renderer/OptixRenderer.h"
#include <QtAlgorithms>

RenderResultPacketReceiver::RenderResultPacketReceiver(const DistributedApplication & application)
    : m_application(application),
      m_frontBuffer(NULL),
      m_backBuffer(NULL),
      m_PPMNextExpectedIteration(0),
      m_lastSequenceNumber(0),
      m_peakBackBufferSizeBytes(0)
{
    m_frontBuffer = new float[2000*2000*3];
}

RenderResultPacketReceiver::~RenderResultPacketReceiver(void)
{
    delete[] m_frontBuffer;
}

// Take ownership of the RenderResultPacket object and merge in the result into
// output buffers

void RenderResultPacketReceiver::onRenderResultPacketReceived(RenderResultPacket* result)
{
    // Merge the results if the sequence number of RenderResultPacket matches the current sequence number    

    if(result->getSequenceNumber() == m_application.getSequenceNumber())
    {
        if(result->getSequenceNumber() > m_lastSequenceNumber)
        {
            resetInternals();
            m_lastSequenceNumber = result->getSequenceNumber();
        }

        if(m_application.getRenderMethod() == RenderMethod::PROGRESSIVE_PHOTON_MAPPING)
        {
            
            mergeRenderResultPacketPhotonMapping(result);
        }
        else
        {
            mergeRenderResultPathTracing(result);
        }

        emit newFrameReadyForDisplay(m_frontBuffer, m_iterationNumber);
    }

    // We take ownership of the result and make sure to delete it
    delete result;

}

void RenderResultPacketReceiver::onThreadStarted()
{
    
}

void RenderResultPacketReceiver::mergeRenderResultPacketPhotonMapping(const RenderResultPacket* packet)
{
    QVector<unsigned long long> iterationsInPacket = packet->getIterationNumbersInPacket();
    emit packetReceived(packet->getSequenceNumber(), iterationsInPacket.size());
    
    unsigned int numPixels = packet->getOutput().size()/sizeof(float);

    m_backBufferMutex.lock();

    m_backBuffer.append(*packet);
    qSort(m_backBuffer);

    // Try to combine any two sequential back buffers

    QVector<RenderResultPacket>::iterator it;
    while(true)
    {
        for(it = m_backBuffer.begin(); it < m_backBuffer.end()-1; it++)
        {
            if(it->getLastIterationNumber() + 1 == (it+1)->getFirstIterationNumber())
            {
                /*printf("%% Merge %d [", it->getFirstIterationNumber());
                for(int i = 0; i < it->getNumIterationsInPacket(); i++)
                {
                    printf("%d ", it->getIterationNumbersInPacket().at(i));
                }

                printf("] with %d [", (it+1)->getFirstIterationNumber());

                for(int i = 0; i < (it+1)->getNumIterationsInPacket(); i++)
                {
                    printf("%d ", (it+1)->getIterationNumbersInPacket().at(i));
                } 
                printf("] in backbuffer\n");*/

                it->merge(*(it+1));
                m_backBuffer.erase(it+1);
                break;
            }
        }
        if(it == m_backBuffer.end()-1)
        {
            break;
        }
    }

    if(m_backBuffer.first().getFirstIterationNumber() == m_PPMNextExpectedIteration)
    {
        RenderResultPacket backBuffer = m_backBuffer.first();
        /*printf("Merge backbuffer ", m_backBuffer.size());
        for(int i = 0; i < backBuffer.getNumIterationsInPacket(); i++)
        {
            printf("%d ", backBuffer.getIterationNumbersInPacket().at(i));
        }*/
            
        mergeBufferRunningAverage((const float*)backBuffer.getOutput().constData(), backBuffer.getNumIterationsInPacket(), 
            m_frontBuffer, m_PPMNextExpectedIteration, numPixels );
        m_PPMNextExpectedIteration = backBuffer.getLastIterationNumber()+1;
        m_backBuffer.pop_front();
        //printf(" into front. Next it: %d", m_PPMNextExpectedIteration);
    }

    /*printf("\nBackbuffer size: %d, expecting %d\n", m_backBuffer.size(), m_PPMNextExpectedIteration);
    for(int i = 0; i < m_backBuffer.size(); i++)
    {
        printf("   %d: [", i);
        for(int j= 0; j < m_backBuffer.at(i).getNumIterationsInPacket(); j++)
        {
            printf("%d ", m_backBuffer.at(i).getIterationNumbersInPacket().at(j));
        }
        printf("]\n");
    }*/

    m_backBufferMutex.unlock();

    if(getBackBufferSizeBytes() > m_peakBackBufferSizeBytes)
    {
        m_peakBackBufferSizeBytes = getBackBufferSizeBytes();
    }

    m_iterationNumber = m_PPMNextExpectedIteration > 0 ? m_PPMNextExpectedIteration-1 : 0;
}

// In path tracing, we simply control a running average of the frames we have received.
// We do not really care about the order of which we receive RenderResultPackets

void RenderResultPacketReceiver::mergeRenderResultPathTracing(const RenderResultPacket* result )
{
    const QByteArray & packetOutput = result->getOutput();
    float* packetDataFloat = (float*)packetOutput.data();
    unsigned int numElements = packetOutput.size()/sizeof(float);
    mergeBufferRunningAverage(packetDataFloat, result->getNumIterationsInPacket(), m_frontBuffer, m_iterationNumber, numElements);
    m_iterationNumber += result->getNumIterationsInPacket();
}

// 

static __inline float average(const float oldf, const float newf, const float newDivSum)
{
    return oldf + (newf-oldf)*newDivSum;
}

void RenderResultPacketReceiver::mergeBufferRunningAverage( const float* inputBuffer, unsigned int inputBufferNumIterations, 
                                                              float* outputBuffer, unsigned int outputBufferNumIterations, 
                                                              unsigned int numPixels )
{
    //printf("\nMerge %d with %d\n", inputBufferNumIterations, outputBufferNumIterations);
    if(outputBufferNumIterations == 0)
    {
        for(unsigned int i = 0; i < numPixels; i += 3)
        {
            outputBuffer[i]   = inputBuffer[i];
            outputBuffer[i+1] = inputBuffer[i+1];
            outputBuffer[i+2] = inputBuffer[i+2];
        }
    }
    else
    {
        float newNumIterationsInOutBuffer = float(outputBufferNumIterations+inputBufferNumIterations);

        for(unsigned int i = 0; i < numPixels; i += 3)
        {
            outputBuffer[i] = average(outputBuffer[i], inputBuffer[i], inputBufferNumIterations/newNumIterationsInOutBuffer);
            outputBuffer[i+1] = average(outputBuffer[i+1], inputBuffer[i+1], inputBufferNumIterations/newNumIterationsInOutBuffer);
            outputBuffer[i+2] = average(outputBuffer[i+2], inputBuffer[i+2], inputBufferNumIterations/newNumIterationsInOutBuffer);
        }
    }
}

// Check if this SORTED input vector consists of sequential iterations, i.e. 0 1 2

bool RenderResultPacketReceiver::iterationNumbersAreSequential(const QVector<unsigned long long> iterationsInPacketSorted )
{
    return (iterationsInPacketSorted.first() + (unsigned long long)iterationsInPacketSorted.size() == iterationsInPacketSorted.last() + 1);
}

unsigned long long RenderResultPacketReceiver::getIterationNumber() const
{
    return m_iterationNumber;
}

// Set a hard number on the max iterations in back buffer
// When exceeded, no new render commands are issued
// If our last sequence number is old, we allow new render requests 

bool RenderResultPacketReceiver::backBufferIsNotFilled() 
{
    return m_lastSequenceNumber != m_application.getSequenceNumber() || getBackBufferNumIterations() < 100;
}

unsigned int RenderResultPacketReceiver::getBackBufferNumIterations()
{
    m_backBufferMutex.lock();
    unsigned int numIterations = 0;
    for(int i = 0; i < m_backBuffer.size(); i++)
    {
        numIterations += m_backBuffer.at(i).getNumIterationsInPacket();
    }
    m_backBufferMutex.unlock();
    return numIterations;
}

unsigned int RenderResultPacketReceiver::getBackBufferSizeBytes()
{
    unsigned int sizeBytes = 0;
    m_backBufferMutex.lock();
    for(int i = 0; i < m_backBuffer.size(); i++)
    {
        sizeBytes += m_backBuffer.at(i).getOutput().size() + sizeof(RenderResultPacket);
    }
    m_backBufferMutex.unlock();
    return sizeBytes;
}

unsigned int RenderResultPacketReceiver::getPeakBackBufferSizeBytes() const
{
    return m_peakBackBufferSizeBytes;
}

void RenderResultPacketReceiver::resetInternals()
{
    m_PPMNextExpectedIteration = 0;
    m_iterationNumber = 0;
    m_backBufferMutex.lock();
    m_backBuffer.clear();
    m_backBufferMutex.unlock();
    m_peakBackBufferSizeBytes = 0;
}
