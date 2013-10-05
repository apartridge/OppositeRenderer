/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "RenderServerConnection.hxx"
#include "commands/ServerCommand.h"
#include "commands/GetServerDetailsCommand.h"
#include <QTimer>

/*
A RenderServerConnection represents a connection to a render server. Each RSC lives in its own thread. Thread managing is done by
RenderServerConnections.
A RSC will send render frame commands to its connected server, and it will receive results from the server which are then passed on
to the RenderResultPacketReceiver.
*/

RenderServerConnection::RenderServerConnection(DistributedApplication & application, QTcpSocket* socket, QString computeDeviceName ) :
    QObject(),
    m_application(application),
    m_socket(socket),
    m_socketDataStream(socket),
    m_serverIp(socket->peerName()),
    m_serverPort(QString::number(socket->peerPort())),
    m_renderServerState(RenderServerState::NO_DEVICE_INFORMATION),
    m_currentCommand(NULL),
    m_renderTimeSeconds(0.0f),
    m_computeDeviceName(computeDeviceName),
    m_expectingSizeOfRenderCommandResult(0),
    m_numServerPendingIterations(0),
    m_lastRenderCommandSequenceNumber(0),
    m_numSentRenderCommands(0),
    m_numIterationsReceived(0),
    m_bytesReceived(0),
    m_numPacketsReceived(0),
    m_pendingIterationsLimit(30),
    m_initialMaxIterationsPerPacket(4),
    m_averageRequestResponseTime(0),
    m_maxIterationsPerPacket(m_initialMaxIterationsPerPacket)
{
    m_socketDataStream.setFloatingPointPrecision(QDataStream::SinglePrecision);
    connect(m_socket, SIGNAL(disconnected()), this, SLOT(onSocketDisconnected()));
    connect(m_socket, SIGNAL(readyRead()), this, SLOT(onSocketReadyRead()));
    connect(this, SIGNAL(newServerCommand(ServerCommand*)), this, SLOT(onNewServerCommand(ServerCommand*)), Qt::QueuedConnection);
   
    m_sendNewRenderCommandTimer = new QTimer();
    m_sendNewRenderCommandTimer->setInterval(300);
    connect(m_sendNewRenderCommandTimer, SIGNAL(timeout()), this, SLOT(onTimeout()));
    m_sendNewRenderCommandTimer->start();
    m_totalTime.restart();
}

RenderServerConnection::~RenderServerConnection()
{
    delete m_currentCommand;
    m_currentCommand = NULL;
    delete m_sendNewRenderCommandTimer;
}

void RenderServerConnection::onAboutToQuit()
{
    printf("RenderServerConnection::onAboutToQuit\n");
    m_socket->disconnect();
    m_socket->deleteLater();
    printf("end RenderServerConnection::onAboutToQuit\n");
}

void RenderServerConnection::pushCommandAsync( ServerCommand* command )
{
    emit newServerCommand(command);
    setRenderServerState(command->getInitialRenderServerState());
}

void RenderServerConnection::onNewServerCommand( ServerCommand* command )
{
    if(m_currentCommand == NULL)
    {
        m_currentCommand = command;
        command->executeCommand(*m_socket);
    }
    else
    {
        throw new std::exception("Cannot call pushCommand while a command is pending on this connection.");
    }
}

void RenderServerConnection::onThreadStarted()
{

}

static int min(int a, int b)
{
    return a < b ? a : b;
}

static int max(int a, int b)
{
    return a > b ? a : b;
}
void RenderServerConnection::onTimeout()
{
    if(getRenderServerState() == RenderServerState::NO_DEVICE_INFORMATION)
    {
        GetServerDetailsCommand* command = new GetServerDetailsCommand();
        pushCommandAsync(command);
    }
    // Keep sending RenderCommands to server until we have reached m_numPendingRenderFrames >= X. 
    // Always send RenderCommand if we have increased sequence number. The RenderServer can then drop rendering of old frames with old sequenceNumber
    else
    {
        bool applicationAndServerRunning = getRenderServerState() == RenderServerState::RENDERING 
             && m_application.getRunningStatus() == RunningStatus::RUNNING;

        bool allowSendNewRequest = (m_numServerPendingIterations < m_pendingIterationsLimit || m_application.getSequenceNumber() > m_lastRenderCommandSequenceNumber );
        if(applicationAndServerRunning && m_application.canIssueNewRenderRequests() && allowSendNewRequest)
        {
            if(m_application.getSequenceNumber() > m_lastRenderCommandSequenceNumber)
            {
                resetInternalStatistics();
            }

            unsigned int numIterationsInRequest = (unsigned int)max(1, m_maxIterationsPerPacket);
            RenderServerRenderRequest request = m_application.getNextRenderServerRenderRequest(numIterationsInRequest);

            if(request.getIterationNumbers().size() > 0)
            {
                m_socketDataStream << request;
                m_lastRenderCommandSequenceNumber = request.getSequenceNumber();
                m_numSentRenderCommands++;
                m_numServerPendingIterations += request.getNumIterations();
                m_pendingIterationSendTimes[request.getFirstIterationNumber()] = getTotalTimeSeconds();
                emit stateUpdated();
                m_socket->flush();
            }
        }
    }
}

void RenderServerConnection::setRenderServerState( RenderServerState::E renderServerState )
{
    q_renderServerStateMutex.lock();
    m_renderServerState = renderServerState;
    emit newRenderServerState(renderServerState);
    q_renderServerStateMutex.unlock();
}

void RenderServerConnection::onSocketDisconnected()
{
    setRenderServerState(RenderServerState::DISCONNECTED);
}

void RenderServerConnection::onSocketError()
{
    setRenderServerState(RenderServerState::ERROR_SOCKET);
}

void RenderServerConnection::onSocketReadyRead()
{
    //qDebug() << "Port: " << m_serverPort << "Got " << m_socket->bytesAvailable() << "bytes!\n";
    if(m_currentCommand != NULL)
    {
        m_expectingSizeOfRenderCommandResult = 0;
        handleCommandResponse();
    }
        
    qint64 bytes_available = m_socket->bytesAvailable();
    
    if(bytes_available > 0)
    {
        if(m_renderServerState == RenderServerState::RENDERING)
        {
            if(m_expectingSizeOfRenderCommandResult == 0)
            {
                // Digest the first 64 bits which contain the expected size of the entire packet
                m_socketDataStream >> m_expectingSizeOfRenderCommandResult;
                bytes_available -= sizeof(m_expectingSizeOfRenderCommandResult);
            }

            if(bytes_available >= m_expectingSizeOfRenderCommandResult)
            {
                RenderResultPacket* result = this->getArrivedRenderCommandResult();
                unsigned long sequenceNumber = result->getSequenceNumber();
                int numIterationsInPacket = result->getNumIterationsInPacket();
                if(result->getSequenceNumber() == m_application.getSequenceNumber())
                {
                    m_bytesReceived += m_expectingSizeOfRenderCommandResult;
                    m_numPacketsReceived += 1;
                    m_numIterationsReceived += result->getNumIterationsInPacket();
                    m_renderTimeSeconds = result->getRenderTimeSeconds();
                    float timeSinceIterationSent = getTotalTimeSeconds() - m_pendingIterationSendTimes[result->getFirstIterationNumber()];
                    addToAverageRequestResponseTime(timeSinceIterationSent);
                    m_pendingIterationSendTimes.remove(result->getFirstIterationNumber());

                    /*// Increase or decrease max iterations per packet based on performance
                    if(timeSinceIterationSent > 5.0)
                    {
                        if(m_maxIterationsPerPacket < 8)
                        {
                            m_maxIterationsPerPacket++;
                        }
                    }
                    else if(m_maxIterationsPerPacket > m_initialMaxIterationsPerPacket)
                    {
                        m_maxIterationsPerPacket--;
                    }*/
                    emit renderResultPacketReceived(result);
                }
                else
                {
                    delete result;
                }
                
                m_expectingSizeOfRenderCommandResult = 0;

                if(sequenceNumber == m_application.getSequenceNumber())
                {
                    m_numServerPendingIterations -= (unsigned int)numIterationsInPacket;
                }
            }
        }
    }

    emit stateUpdated();
    //bytes_available = m_socket->bytesAvailable();
    //printf("\tBytes available after : %d\n", bytes_available);
}

void RenderServerConnection::handleCommandResponse()
{
    if(m_currentCommand != NULL)
    {
        //qDebug() << "Port: " << m_serverPort; 
        ServerCommandResult result = m_currentCommand->onResponseReady(*this, *m_socket);
        setRenderServerState(result.getNewRenderServerState());
        delete m_currentCommand;
        m_currentCommand = NULL;
    }
    // else no pending command
}

RenderResultPacket* RenderServerConnection::getArrivedRenderCommandResult()
{
    RenderResultPacket* result = new RenderResultPacket();
    m_socketDataStream >> *result;
    return result;
}

void RenderServerConnection::resetInternalStatistics()
{
    m_numServerPendingIterations = 0;
    m_numIterationsReceived = 0;
    m_numPacketsReceived = 0;
    m_numSentRenderCommands = 0;
    m_bytesReceived = 0;
    m_maxIterationsPerPacket = m_initialMaxIterationsPerPacket;
    m_totalTime.restart();
    m_renderTimeSeconds = 0;
    m_averageRequestResponseTime = 0;
    //m_pendingIterationSendTimes.clear();
}

float RenderServerConnection::getRenderTimeSeconds() const
{
    return m_renderTimeSeconds;
}

float RenderServerConnection::getTotalTimeSeconds() const
{
    return m_totalTime.elapsed()/1000.0f;
}

float RenderServerConnection::getServerEfficiency() const
{
    return getTotalTimeSeconds() > 0 ? m_renderTimeSeconds/getTotalTimeSeconds() : 0;
}

const QString & RenderServerConnection::getComputeDeviceName() const
{
    return m_computeDeviceName;
}

void RenderServerConnection::moveSocketToThread( QThread* thread )
{
    m_socket->moveToThread(thread);
}

void RenderServerConnection::setComputeDeviceName( QString name )
{
    m_computeDeviceName = name;
}

unsigned int RenderServerConnection::getNumPendingIterations() const
{
    return m_numServerPendingIterations;
}

unsigned long long RenderServerConnection::getBytesReceived() const
{
    return m_bytesReceived;
}

unsigned long long RenderServerConnection::getNumPacketsReceived() const
{
    return m_numPacketsReceived;
}

const QString RenderServerConnection::getServerIp() const
{
    return m_serverIp;
}

const QString RenderServerConnection::getServerPort() const
{
    return m_serverPort;
}

const QString RenderServerConnection::getConnectionState() const
{
    switch(m_socket->state())
    {
    case QAbstractSocket::ConnectedState: return "Connected";
    case QAbstractSocket::UnconnectedState: return "Not connected";
    }
    return m_serverPort;
}

RenderServerState::E RenderServerConnection::getRenderServerState() const
{
    return m_renderServerState;
}

unsigned long long RenderServerConnection::getNumIterationsReceived() const
{
    return m_numIterationsReceived;
}

unsigned int RenderServerConnection::getPendingIterationsLimit() const
{
    return m_pendingIterationsLimit;
}

unsigned int RenderServerConnection::getMaxIterationsPerPacket() const
{
    return m_maxIterationsPerPacket;
}

void RenderServerConnection::addToAverageRequestResponseTime( float latency )
{
    m_averageRequestResponseTime = (m_averageRequestResponseTime*(m_numPacketsReceived-1) + latency)/m_numPacketsReceived; 
}

float RenderServerConnection::getAverageRequestResponseTime() const
{
    return m_averageRequestResponseTime;
}
