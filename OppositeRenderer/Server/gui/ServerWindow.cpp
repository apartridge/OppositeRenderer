/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "ServerWindow.hxx"
#include "ui/ui_ServerWindow.h"
#include <QTcpServer>
#include <QTcpSocket>
#include <QMessageBox>
#include <QLabel>
#include "gui/ComputeDeviceInformationWidget.hxx"
#include "ComputeDeviceRepository.h"
#include "ComputeDevice.h"
#include "gui/AboutWindow.hxx"
#include "gui/SetServerSettingsWidget.hxx"
#include "gui/WaitingForConnectionWidget.hxx"
#include "gui/ReadyForRenderingWidget.hxx"
#include "server/RenderServer.hxx"
#include <QTimer>

ServerWindow::ServerWindow(QWidget *parent, RenderServer & serverApplication) :
    QMainWindow(parent),
    ui(new Ui::ServerWindow),
    m_clientSocket(NULL),
    m_server(NULL),
    m_computeDevice(NULL),
    m_serverState(ServerState::SET_COMPUTE_DEVICE),
    m_renderServer(serverApplication)
{
    // UI STUFF
    ui->setupUi(this);
    this->m_serverStateLabel = new QLabel(this);
    this->m_renderStateLabel = new QLabel(this);
    ui->statusbar->addPermanentWidget(this->m_serverStateLabel, 2);
    ui->statusbar->addPermanentWidget(this->m_renderStateLabel, 3);
    QFrame* frame = new QFrame(this->centralWidget());
    m_setComputeDeviceWidget = new ComputeDeviceInformationWidget(frame, m_computeDeviceRepository);
    connect(m_setComputeDeviceWidget, SIGNAL(hasSelectedComputeDevice(ComputeDevice*)), this, SLOT(onHasSelectedComputeDevice(ComputeDevice*)));
    frame->setGeometry(0,0,540,370);
    m_setComputeDeviceWidget->hide();
    this->setWindowTitle("RenderServer");

    m_serverSettingsWidget = new SetServerSettingsWidget(this->centralWidget());
    m_serverSettingsWidget->hide();

    m_waitingForConnectionWidget = new WaitingForConnectionWidget(this->centralWidget());
    m_waitingForConnectionWidget->hide();

    m_readyForRenderingWidget = new ReadyForRenderingWidget(this->centralWidget());
    m_readyForRenderingWidget->hide();

    // SERVER STUFF
    connect(this, SIGNAL(newServerState(ServerState::E)), this, SLOT(onNewServerState(ServerState::E)));
    connect(&m_renderServer, SIGNAL(renderStateUpdated(RenderServerState::E)), this, SLOT(onNewRenderState(RenderServerState::E)));
    connect(&m_renderServer, SIGNAL(logStringAppended(QString)), this, SLOT(onNewServerApplicationLogString(QString)));
    
    QTimer* timer = new QTimer(this);
    timer->setInterval(150);
    connect(timer, SIGNAL(timeout()), this, SLOT(onTimeout()));
    timer->start();

    setServerState(ServerState::SET_COMPUTE_DEVICE);
}

ServerWindow::~ServerWindow()
{
    delete ui;

    if(m_clientSocket)
    {
        delete m_clientSocket;
        m_clientSocket = NULL;
    }

    // m_computeDevice is handled by the ComputeDeviceRepository
    // m_server is a child of this and handled by Qt
}

// Reset the process to the point of selecting compute device. 

void ServerWindow::resetProcess()
{
    if(m_clientSocket != NULL)
    {
        if(m_clientSocket->state() == QAbstractSocket::ConnectedState)
        {
            m_clientSocket->disconnectFromHost();
            if(m_clientSocket->state() == QAbstractSocket::ConnectedState)
            {
                m_clientSocket->waitForDisconnected();
            }
        }
        m_clientSocket = NULL;
    }

    if(m_server != NULL && m_server->isListening())
    {
        m_server->close();
    }

    if(m_computeDevice != NULL)
    {
        delete m_computeDevice;
        m_computeDevice = NULL;
    }

    setServerState(ServerState::SET_COMPUTE_DEVICE);
}

/*
// Select compute device
*/

void ServerWindow::onStateSetComputeDeviceEnter()
{
    m_setComputeDeviceWidget->show();
}

void ServerWindow::onStateSetComputeDeviceExit()
{
    m_setComputeDeviceWidget->hide();
}

void ServerWindow::onHasSelectedComputeDevice( ComputeDevice* device )
{
    m_computeDevice = device;
    m_renderServer.initializeDevice(*m_computeDevice);
    setServerState(ServerState::SET_SERVER_SETTINGS);
}

/*
// Configure server port
*/

void ServerWindow::onStateSetServerSettingsEnter( )
{
    this->setWindowTitle(QString("RenderServer [#%1 - %2]").arg(m_computeDevice->getDeviceId())
                                                        .arg(m_computeDevice->getName()));
    m_serverSettingsWidget->show();
    connect(m_serverSettingsWidget, SIGNAL(startServerFormSubmitted()), this, SLOT(onStartServerFormSubmitted()));
}

void ServerWindow::onStartServerFormSubmitted()
{
    m_serverPort = m_serverSettingsWidget->getPortNumber();
    
    if(m_server == NULL)
    {
        m_server = new QTcpServer(this);
        connect(m_server, SIGNAL(newConnection()), this, SLOT(onNewConnection()));
    }
    else
    {
        m_server->close();
    }

    if (!m_server->listen(QHostAddress::Any, m_serverPort))
    {
        QMessageBox::critical(this, "Server",
            QString("Unable to start the server on port %1: %2.")
            .arg(m_server->errorString()).arg(m_serverPort));
    }
    else
    {
        setServerState(ServerState::WAIT_FOR_CLIENT_CONNECTION);
    }
}

void ServerWindow::onStateSetServerSettingsExit( )
{
    m_serverSettingsWidget->hide();
}

/*
// Wait for connection. Set the server to listen.
*/

void ServerWindow::onStateWaitForConnectionEnter()
{
    m_waitingForConnectionWidget->setPortNumber(QString::number(m_serverPort));
    m_waitingForConnectionWidget->show();
    m_renderStateLabel->setText("");
    m_readyForRenderingWidget->clearLog();
}

void ServerWindow::onStateWaitForConnectionExit()
{
    m_waitingForConnectionWidget->hide();
}

void ServerWindow::onNewConnection()
{
    if(m_serverState == ServerState::WAIT_FOR_CLIENT_CONNECTION)
    {
        m_clientSocket = m_server->nextPendingConnection();

        QString computeDeviceName = QString("RSHELLO\n");
        QDataStream stream (m_clientSocket);
        stream << computeDeviceName;

        connect(m_clientSocket, SIGNAL(disconnected()), this, SLOT(onClientConnectionDisconnected()));
        setServerState(ServerState::READY_FOR_RENDERING);
    }
    else
    {
        QTcpSocket* socket = m_server->nextPendingConnection();
        socket->write("E Busy; connected to a client!");
        socket->close();
        socket->deleteLater();
        printf("A new connection to server was ignored (is connected to a client)!\n");
    }
}

void ServerWindow::onClientConnectionDisconnected()
{
    m_clientSocket = NULL;
    setServerState(ServerState::WAIT_FOR_CLIENT_CONNECTION);
}

/*
// Ready status
*/

void ServerWindow::onStateReadyForRenderingEnter()
{
    // Connection info
    QHostAddress clientAddress = m_clientSocket->peerAddress();
    quint16 clientPort = m_clientSocket->peerPort();
    m_readyForRenderingWidget->setComputeDevice(*m_computeDevice);
    m_readyForRenderingWidget->setClientName(QString("%1:%2").arg(clientAddress.toString(), QString::number(clientPort)));

    QHostAddress serverAddress = m_server->serverAddress();
    quint16 serverPort = m_server->serverPort();
    m_readyForRenderingWidget->setServerName(QString("%1:%2").arg(serverAddress.toString(), QString::number(serverPort)));
    m_readyForRenderingWidget->show();

    // Initialize Render Manager with selected compute device and socket

    m_renderServer.initializeClient(*m_clientSocket);
}

void ServerWindow::onStateReadyForRenderingExit()
{
    m_readyForRenderingWidget->hide();
}

void ServerWindow::setServerState(ServerState::E state)
{
    switch(m_serverState)
    {
    case ServerState::WAIT_FOR_CLIENT_CONNECTION: onStateWaitForConnectionExit(); break;
    case ServerState::SET_COMPUTE_DEVICE: onStateSetComputeDeviceExit(); break;
    case ServerState::SET_SERVER_SETTINGS: onStateSetServerSettingsExit(); break;
    case ServerState::READY_FOR_RENDERING: onStateReadyForRenderingExit(); break;
    }

    m_serverState = state;

    switch(m_serverState)
    {
    case ServerState::WAIT_FOR_CLIENT_CONNECTION: onStateWaitForConnectionEnter(); break;
    case ServerState::SET_COMPUTE_DEVICE: onStateSetComputeDeviceEnter(); break;
    case ServerState::SET_SERVER_SETTINGS: onStateSetServerSettingsEnter(); break;
    case ServerState::READY_FOR_RENDERING: onStateReadyForRenderingEnter(); break;
    }

    emit newServerState(state);
}

/*
UI RELATED
*/

void ServerWindow::onNewServerState(ServerState::E serverState)
{
    this->m_serverStateLabel->setText(QString(serverStateEnumToString(serverState)));
}

void ServerWindow::onActionAbout()
{
    AboutWindow* ab = new AboutWindow(this);
    ab->show();
}

void ServerWindow::onActionSetComputeDevice()
{
    resetProcess();
}

void ServerWindow::onNewRenderState(RenderServerState::E renderState)
{
    this->m_renderStateLabel->setText(QString(renderStateEnumToText(renderState)));
}

void ServerWindow::onNewServerApplicationLogString(QString string)
{
    m_readyForRenderingWidget->appendToLog(string);
}

void ServerWindow::onTimeout()
{
    m_readyForRenderingWidget->setRenderTime(m_renderServer.getRenderTimeSeconds(), m_renderServer.getTotalTimeSeconds());
    m_readyForRenderingWidget->setPendingRenderCommandsAndIterations(m_renderServer.getNumPendingRenderIterations(), 
                                                                     m_renderServer.getNumPendingRenderCommands());
}
