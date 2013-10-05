/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "DistributedApplication.hxx"
#include "ComputeDevice.h"
#include <QApplication>
#include <QObject>
#include <QTime>
#include "RunningStatus.h"
#include "renderer/RenderMethod.h"
#include "renderer/Camera.h"
#include "Application.hxx"
#include "client/RenderServerConnections.hxx"
#include "client/RenderServerConnection.hxx"
#include "clientserver/RenderServerRenderRequest.h"
#include "client/RenderResultPacketReceiver.hxx"
#include <QMutex>
#include "renderer/OptixRenderer.h"
#include <QThread>

DistributedApplication::DistributedApplication(QApplication & qApplication)
    : Application(qApplication),
    m_renderResultPacketReceiver(RenderResultPacketReceiver(*this)),
    m_nextRenderServerRenderRequestIteration(0),
    m_lastSequenceNumber(0),
    m_totalPacketsPending(0),
    m_numPreviewedIterations(0),
    m_totalPacketsPendingLimit(80)
{
    m_renderResultPacketReceiverThread = new QThread(this);
    m_renderResultPacketReceiver.moveToThread(m_renderResultPacketReceiverThread);
    connect(m_renderResultPacketReceiverThread, SIGNAL(started()), &m_renderResultPacketReceiver, SLOT(onThreadStarted()));
    m_renderResultPacketReceiverThread->start(QThread::HighPriority);

    connect(this, SIGNAL(sequenceNumberIncremented()), 
            this, SLOT(onSequenceNumberIncremented()), 
            Qt::QueuedConnection);

    connect(&m_renderResultPacketReceiver, SIGNAL(newFrameReadyForDisplay(const float*, unsigned long long)), 
            this, SIGNAL(newFrameReadyForDisplay(const float*, unsigned long long)), 
            Qt::QueuedConnection);

    connect(&m_renderResultPacketReceiver, SIGNAL(newFrameReadyForDisplay(const float*, unsigned long long)), 
            this, SLOT(onNewFrameReadyForDisplay(const float*, unsigned long long)));

    connect(&m_renderResultPacketReceiver, SIGNAL(packetReceived(unsigned long long, unsigned int)), 
        this, SLOT(onPacketReceived(unsigned long long, unsigned int)));

    setRendererStatus(RendererStatus::RENDERING);
}

DistributedApplication::~DistributedApplication(void)
{

}

void DistributedApplication::onAboutToQuit()
{
    m_renderResultPacketReceiverThread->quit();
}

void DistributedApplication::wait()
{
    printf("DistributedApplication::wait\n");
    waitOnApplicationFinished();
    m_renderResultPacketReceiverThread->wait();
}

void DistributedApplication::onThreadStarted()
{

}

void DistributedApplication::onNewServerConnectionSocket(QTcpSocket* socket)
{
    RenderServerConnection* connection = new RenderServerConnection(*this, socket, QString(""));

    // Connect the RSC and the RenderResultPacketReceiver so that RFR will receive RenderCommandResults
    // from all server connections and merge the results

    connect(connection, SIGNAL(renderResultPacketReceived(RenderResultPacket*)), 
        &m_renderResultPacketReceiver, SLOT(onRenderResultPacketReceived(RenderResultPacket*)),
        Qt::QueuedConnection);

    m_serverConnections.push_back(connection);
}

const RenderServerConnections & DistributedApplication::getServerConnections() const
{
    return m_serverConnections;
}

RenderServerRenderRequest DistributedApplication::getNextRenderServerRenderRequest(unsigned int numIterations)
{
    QVector<unsigned long long> iterationNumbers;
    QVector<double> ppmRadii;

    double PPMAlpha = 2.0/3.0;

    m_mutex.lock();

    for(int i = 0; i < numIterations; i++)
    {
        iterationNumbers.push_back(m_nextRenderServerRenderRequestIteration);
        ppmRadii.push_back(m_PPMRadius);

        double ppmRadiusSq = m_PPMRadius*m_PPMRadius;
        double ppmRadiusSqNew = ppmRadiusSq*(m_nextRenderServerRenderRequestIteration+PPMAlpha)/(m_nextRenderServerRenderRequestIteration+1);
        m_PPMRadius = sqrt(ppmRadiusSqNew);
        m_nextRenderServerRenderRequestIteration++;
    }

    QByteArray sceneName = QByteArray(getSceneManager().getScene()->getSceneName());
    RenderServerRenderRequestDetails details (getCamera(), sceneName, getRenderMethod(), getOutputSettingsModel().getWidth(), getOutputSettingsModel().getHeight(), PPMAlpha);
    RenderServerRenderRequest request (getSequenceNumber(), iterationNumbers, ppmRadii, details);
    m_totalPacketsPending++;
    m_mutex.unlock();
    return request;
}

void DistributedApplication::onNewFrameReadyForDisplay(const float*, unsigned long long iterationNumber)
{
    m_numPreviewedIterations++;
    getRenderStatisticsModel().setNumIterations(iterationNumber+1);
    getRenderStatisticsModel().setCurrentPPMRadius(m_PPMRadius);
    getRenderStatisticsModel().setNumPreviewedIterations(m_numPreviewedIterations);

    if(getRenderMethod() == RenderMethod::PROGRESSIVE_PHOTON_MAPPING)
    {
        getRenderStatisticsModel().setNumEmittedPhotonsPerIteration(OptixRenderer::EMITTED_PHOTONS_PER_ITERATION);
        getRenderStatisticsModel().setNumEmittedPhotons(OptixRenderer::EMITTED_PHOTONS_PER_ITERATION*(iterationNumber+1));
    }
    else
    {
        getRenderStatisticsModel().setNumEmittedPhotonsPerIteration(0);
        getRenderStatisticsModel().setNumEmittedPhotons(0);
    }
}

void DistributedApplication::onSequenceNumberIncremented()
{
    m_mutex.lock();
    m_nextRenderServerRenderRequestIteration = 0;
    m_totalPacketsPending = 0;
    m_numPreviewedIterations = 0;
    m_PPMRadius = getPPMSettingsModel().getPPMInitialRadius();
    m_mutex.unlock();
}

bool DistributedApplication::canIssueNewRenderRequests()
{
    return m_totalPacketsPending < m_totalPacketsPendingLimit && m_renderResultPacketReceiver.backBufferIsNotFilled();
}

unsigned int DistributedApplication::getBackBufferNumIterations()
{
    return m_renderResultPacketReceiver.getBackBufferNumIterations();
}

unsigned int DistributedApplication::getBackBufferSizeBytes() 
{
    return m_renderResultPacketReceiver.getBackBufferSizeBytes();
}

unsigned int DistributedApplication::getPeakBackBufferSizeBytes() const
{
    return m_renderResultPacketReceiver.getPeakBackBufferSizeBytes();
}

unsigned int DistributedApplication::getTotalPacketsPending() const
{
    return m_totalPacketsPending;
}

void DistributedApplication::onPacketReceived( unsigned long long sequenceNumber, unsigned int numIterations )
{
    if(sequenceNumber == getSequenceNumber() && m_totalPacketsPending > 0)
    {
        m_totalPacketsPending--;
    }
}
