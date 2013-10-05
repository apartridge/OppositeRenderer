/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include <vector>
#include <QString>
#include <QTcpSocket>
#include "RenderServerState.h"
#include "DistributedApplication.hxx"
#include "clientserver/RenderResultPacket.h"
#include <QMutex>
#include <QDataStream>
#include <QTime>
#include <QMap>

class ServerCommand;
class DistributedApplication;
class QTimer;

class RenderServerConnection : public QObject
{
    Q_OBJECT;

public:
    // Pass ownership of the socket
    RenderServerConnection(DistributedApplication & renderManager, QTcpSocket* socket, QString computeDeviceName);
    ~RenderServerConnection();
    const QString getServerIp() const;
    const QString getServerPort() const;
    const QString & getComputeDeviceName() const;
    void setComputeDeviceName(QString name);
    const QString getConnectionState() const;
    unsigned long long getNumIterationsReceived() const;
    unsigned long long getBytesReceived() const;
    unsigned long long getNumPacketsReceived() const;
    unsigned int getNumPendingIterations() const;
    unsigned int getPendingIterationsLimit() const;
    unsigned int getMaxIterationsPerPacket() const;
    float getRenderTimeSeconds() const;
    float getTotalTimeSeconds() const;
    float getServerEfficiency() const;
    float getAverageRequestResponseTime() const;

    // Send a command and pass ownership of the command object
    void pushCommandAsync( ServerCommand* command );
    RenderServerState::E getRenderServerState() const;
    void moveSocketToThread( QThread* thread );

public slots:
    void onThreadStarted();
    void onAboutToQuit();
    void onNewServerCommand(ServerCommand*);

signals:
    void newRenderServerState(RenderServerState::E);
    void renderResultPacketReceived(RenderResultPacket*);
    void newServerCommand(ServerCommand*);
    void stateUpdated();

private slots:
    void onSocketDisconnected();
    void onSocketError();
    void onSocketReadyRead();
    void onTimeout();

    void resetInternalStatistics();

private:
    void addToAverageRequestResponseTime(float);
    QTime m_totalTime;
    DistributedApplication & m_application;
    RenderResultPacket* getArrivedRenderCommandResult();
    void setRenderServerState(RenderServerState::E);
    ServerCommand* m_currentCommand;
    QMap<unsigned long long, float> m_pendingIterationSendTimes;
    QTcpSocket* m_socket;
    QDataStream m_socketDataStream;
    QMutex q_renderServerStateMutex;
    QString m_serverIp;
    QString m_serverPort;
    RenderServerState::E m_renderServerState;
    RenderServerConnection(const RenderServerConnection &);
    RenderServerConnection & operator=(const RenderServerConnection &);
    void handleCommandResponse();
    QByteArray receiveBuffer;
    quint64 m_expectingSizeOfRenderCommandResult;
    float m_averageRequestResponseTime;
    unsigned long long m_numSentRenderCommands;

    unsigned int m_numServerPendingIterations;
    unsigned int m_pendingIterationsLimit;
    unsigned int m_maxIterationsPerPacket;
    const unsigned int m_initialMaxIterationsPerPacket;

    unsigned long long m_lastRenderCommandSequenceNumber;
    unsigned long long m_numIterationsReceived;

    unsigned long long m_bytesReceived;
    unsigned long long m_numPacketsReceived;

    float m_renderTimeSeconds;
    QString m_computeDeviceName;
    QTimer* m_sendNewRenderCommandTimer;
};