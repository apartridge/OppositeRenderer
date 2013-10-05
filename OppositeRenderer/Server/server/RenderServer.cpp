/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "RenderServer.hxx"
#include "ComputeDevice.h"
#include <QTcpSocket>
#include "clientserver/RenderServerRenderRequest.h"
#include <QThread>

RenderServer::RenderServer(void)
    : m_renderState(RenderServerState::NOT_VALID_RENDER_STATE),
      m_clientSocket(NULL),
      m_renderServerRenderer(RenderServerRenderer(*this)),
      m_clientSocketDataStream(NULL),
      m_clientExpectingBytes(0)
{
    m_renderServerRendererThread = new QThread();
    m_renderServerRenderer.moveToThread(m_renderServerRendererThread);
    m_renderServerRendererThread->start();

    //connect(this, SIGNAL(aboutToQuit()), m_renderServerRendererThread, SLOT(quit()));

    connect(m_renderServerRendererThread, SIGNAL(started()), &m_renderServerRenderer, SLOT(onThreadStarted()));
    connect(&m_renderServerRenderer, SIGNAL(newLogString(QString)), 
        this, SLOT(appendToLog(QString)), Qt::QueuedConnection);
    connect(&m_renderServerRenderer, SIGNAL(newRenderResultPacket(RenderResultPacket)), 
            this, SLOT(onNewRenderResultPacket(RenderResultPacket)), 
            Qt::QueuedConnection);
}

RenderServer::~RenderServer(void)
{
    delete m_renderServerRendererThread;
    delete m_clientSocketDataStream;
}

void RenderServer::wait()
{
    QMetaObject::invokeMethod(&m_renderServerRenderer, "onAboutToQuit", Qt::QueuedConnection);
    m_renderServerRenderer.wait();
    m_renderServerRendererThread->exit();
    m_renderServerRendererThread->wait();
}

void RenderServer::initializeDevice(const ComputeDevice & computeDevice)
{
    m_renderServerRenderer.initialize(&computeDevice);
}

void RenderServer::initializeClient(QTcpSocket & clientSocket)
{
    m_clientSocket = &clientSocket;
    // Delete old client data stream so that we can initialize multiple times
    delete m_clientSocketDataStream;
    m_clientSocketDataStream = new QDataStream(m_clientSocket);
    m_clientSocketDataStream->setFloatingPointPrecision(QDataStream::SinglePrecision);
    connect(m_clientSocket, SIGNAL(readyRead()), this, SLOT(onDataFromClient()));
    connect(m_clientSocket, SIGNAL(disconnected()), &m_renderServerRenderer, SLOT(onClientDisconnected()));
    setRenderState(RenderServerState::WAITING_FOR_INTRODUCTION_REQUEST);
    m_renderState = RenderServerState::WAITING_FOR_INTRODUCTION_REQUEST;
    m_clientExpectingBytes = 0;
    m_renderServerRenderer.initializeNewClient();
}

void RenderServer::setRenderState(RenderServerState::E renderState)
{
    m_renderState = renderState;
    emit renderStateUpdated(renderState);
}

void RenderServer::onDataFromClient()
{
    if(m_renderState == RenderServerState::WAITING_FOR_INTRODUCTION_REQUEST)
    {
        QByteArray arr = m_clientSocket->readAll();
        const char* dataPtr = arr.constData();
        if(strncmp(dataPtr, "GET SERVER DETAILS", 18) == 0)
        {
            QString computeDeviceName = QString("%1 (#%2, CC %3)")
                .arg(m_renderServerRenderer.getComputeDevice().getName())
                .arg(m_renderServerRenderer.getComputeDevice().getDeviceId())
                .arg(m_renderServerRenderer.getComputeDevice().getComputeCapability());
            *m_clientSocketDataStream << computeDeviceName;
            m_clientSocket->waitForBytesWritten();
            setRenderState(RenderServerState::RENDERING);
            return;
        }
        setRenderState(RenderServerState::ERROR_UNKNOWN);
    }
    else if(m_renderState == RenderServerState::RENDERING)
    {
        while(m_clientSocket->bytesAvailable() > 0)
        {
            quint64 bytesAvailable = m_clientSocket->bytesAvailable();
            if(m_clientExpectingBytes == 0)
            {
                *m_clientSocketDataStream >> m_clientExpectingBytes;
            }

            if(bytesAvailable >= m_clientExpectingBytes)
            {
                m_clientExpectingBytes = 0;
                RenderServerRenderRequest renderRequest = getRenderServerRenderRequestFromClient();
                m_renderServerRenderer.pushCommandToQueue(renderRequest);

                //emit newRenderCommand(renderRequest);
            }
        }
    }
}

RenderServerRenderRequest RenderServer::getRenderServerRenderRequestFromClient()
{
    //m_pendingRenderCommands++;

    RenderServerRenderRequest renderRequest;
    QDataStream datastream(m_clientSocket);
    datastream >> renderRequest;
    QDataStream::Status s = datastream.status();
    return renderRequest;
}

void RenderServer::appendToLog(QString log)
{
    emit logStringAppended(log);
}

void RenderServer::sendConfirmationToClient()
{
    m_clientSocket->write("OK\n");
}

void RenderServer::onNewRenderResultPacket(RenderResultPacket result)
{
    m_iterationsRendered++;

    // Piggyback render and total time for this server on the result
    // object so that the client can see performance measure
    result.setRenderTimeSeconds(getRenderTimeSeconds());
    result.setTotalTimeSeconds(getTotalTimeSeconds());

    //printf("Sending result it %d size %d to client. Pending: %d\n", result.getIterationNumber(), result.getDirectRadiance().size(), m_pendingRenderCommands);

    *m_clientSocketDataStream << result;
    m_clientSocket->flush();
}

double RenderServer::getTotalTimeSeconds()
{
    return m_renderServerRenderer.getTotalTimeSeconds();
}

double RenderServer::getRenderTimeSeconds()
{
    return m_renderServerRenderer.getRenderTimeSeconds();
}

unsigned int RenderServer::getNumPendingRenderCommands()
{
    return m_renderServerRenderer.getNumPendingRenderCommands();
}

unsigned int RenderServer::getNumPendingRenderIterations()
{
    return m_renderServerRenderer.getNumPendingRenderIterations();
}
