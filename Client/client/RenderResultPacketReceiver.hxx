/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include <QObject>
#include <QVector>
#include <QMutex>

/*
This class will receive signals from RenderServerConnections each time the render server has produced a render (given as a 
RenderResultPacket.
This class will do the necessary average/merging of the different subcomputations into the final render, and emits a signal
each time a new frame is ready to be displayed.
*/

class RenderResultPacket;
class DistributedApplication;

class RenderResultPacketReceiver : public QObject
{
    Q_OBJECT;
public:
    RenderResultPacketReceiver(const DistributedApplication & renderManager);
    ~RenderResultPacketReceiver(void);
    unsigned long long getIterationNumber() const;
    bool backBufferIsNotFilled();
    unsigned int getBackBufferNumIterations();
    unsigned int getBackBufferSizeBytes();
    unsigned int getPeakBackBufferSizeBytes() const;

signals:
    void newFrameReadyForDisplay(const float*, unsigned long long);
    void packetReceived(unsigned long long sequenceNumber, unsigned int numIterations);

public slots:
    void onThreadStarted();
    void onRenderResultPacketReceived(RenderResultPacket*);

private:
    const DistributedApplication & m_application;
    unsigned long long m_iterationNumber;
    unsigned long long m_lastSequenceNumber;
    unsigned long long m_PPMNextExpectedIteration;
    unsigned int m_peakBackBufferSizeBytes;
    float* m_frontBuffer;
    void resetInternals();
    //float* m_backBuffer;
    //float* m_indirectPowerPerAreaBackBuffer;
    //QVector<unsigned long long> m_backBufferIterationNumbers;
    QMutex m_backBufferMutex;
    QVector<RenderResultPacket> m_backBuffer;

    void mergeRenderResultPathTracing(const RenderResultPacket* result );
    void mergeRenderResultPacketPhotonMapping(const RenderResultPacket* result );
    void mergeBufferRunningAverage( const float* inputBuffer, unsigned int inputBufferNumIterations, float* outputBuffer, 
            unsigned int outputBufferNumIterations,  unsigned int numPixels );
    bool iterationNumbersAreSequential(const QVector<unsigned long long> iterationsInPacketSorted);
};

