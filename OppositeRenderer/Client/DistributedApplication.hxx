/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once

#include "Application.hxx"
#include <QObject>
#include <QTime>
#include "RunningStatus.h"
#include "client/RenderServerConnections.hxx"
#include "clientserver/RenderServerRenderRequest.h"
#include "client/RenderResultPacketReceiver.hxx"
#include <QMutex>

class QApplication;
class RenderServerConnections;
class QTcpSocket;

class DistributedApplication : public Application
{
    Q_OBJECT;
public:
    DistributedApplication(QApplication & qApplication);
    ~DistributedApplication(void);
    const RenderServerConnections & getServerConnections() const;
    void wait();
    RenderServerRenderRequest getNextRenderServerRenderRequest(unsigned int numIterations);
    bool canIssueNewRenderRequests();
    unsigned int getBackBufferNumIterations();
    unsigned int getTotalPacketsPending() const;
    unsigned int getBackBufferSizeBytes();
    unsigned int getPeakBackBufferSizeBytes() const;

public slots:
    void onThreadStarted();
    void onAboutToQuit();

private slots:
    void onNewServerConnectionSocket(QTcpSocket*);
    void onSequenceNumberIncremented();
    void onNewFrameReadyForDisplay(const float*, unsigned long long);
    void onPacketReceived(unsigned long long sequenceNumber, unsigned int numIterations);
private:
    double m_PPMRadius;
    RenderServerConnections m_serverConnections;
    unsigned long long m_nextRenderServerRenderRequestIteration;
    unsigned long long m_lastSequenceNumber;
    unsigned long long m_numPreviewedIterations;
    unsigned long long m_totalPacketsPending;
    unsigned long long m_totalPacketsPendingLimit;
    RenderResultPacketReceiver m_renderResultPacketReceiver;
    QThread* m_renderResultPacketReceiverThread;
    QMutex m_mutex;
};