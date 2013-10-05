/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

/*
The renderserver responds to RenderServerRenderRequest from the client, and performs them.
The RenderServerRenderer is actually responsible for the rendering, and lives in its own thread
This class deals with network communication between client and server.
*/

#pragma once
#include <QObject>
#include <QString>
#include "RenderServerState.h"
#include "clientserver/RenderResultPacket.h"
#include "clientserver/RenderServerRenderRequest.h"
#include "RenderServerRenderer.hxx"
#include <QDataStream>
#include <QTime>

class ComputeDevice;
class QByteArray;
class IScene;
class QTcpSocket;

class RenderServer : public QObject
{
    Q_OBJECT;
public:
    RenderServer(void);
    ~RenderServer(void);
    void initializeDevice(const ComputeDevice & computeDevice);
    void initializeClient(QTcpSocket & clientSocket);
    unsigned int getNumPendingRenderIterations();
    double getRenderTimeSeconds();
    double getTotalTimeSeconds();
    unsigned int getNumPendingRenderCommands();
    void wait();

public slots:
    void onDataFromClient();
    void sendConfirmationToClient();
    void appendToLog(QString);
    void onNewRenderResultPacket(RenderResultPacket);

signals:
    void renderStateUpdated(RenderServerState::E);
    void logStringAppended(QString);
    void renderTimeUpdated();

private:
    RenderResultPacket getRenderFrameResult(const RenderServerRenderRequest & renderRequest);
    void setRenderState(RenderServerState::E renderState);
    RenderServerRenderRequest getRenderServerRenderRequestFromClient();
    RenderServerState::E m_renderState;

    QTcpSocket* m_clientSocket;
    QDataStream* m_clientSocketDataStream;
    int m_clientExpectingBytes;

    RenderServerRenderer m_renderServerRenderer;
    QThread* m_renderServerRendererThread;
    unsigned long long m_iterationsRendered;
};