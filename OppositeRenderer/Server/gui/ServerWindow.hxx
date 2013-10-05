/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "ServerState.h"
#include "server/RenderServerState.h"
#include "ComputeDeviceRepository.h"

class QTcpSocket;
class QTcpServer;
class QLabel;
class ComputeDevice;
class ComputeDeviceInformationWidget;
class SetServerSettingsWidget;
class WaitingForConnectionWidget;
class ReadyForRenderingWidget;
class RenderServer;

namespace Ui
{
    class ServerWindow;
}

class ServerWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit ServerWindow(QWidget *parent, RenderServer & serverApplication);
    ~ServerWindow();

public slots:
    void onNewConnection();
    void onActionSetComputeDevice();
    void onNewServerState(ServerState::E);
    void onNewServerApplicationLogString(QString);

signals:
    void newServerState(ServerState::E);


private slots:
    void onClientConnectionDisconnected();
    void onStartServerFormSubmitted();
    void onHasSelectedComputeDevice(ComputeDevice*);
    void onActionAbout();
    void onNewRenderState(RenderServerState::E);
    void onTimeout();

private:
    // UI
    Ui::ServerWindow *ui;
    QLabel* m_serverStateLabel;
    QLabel* m_renderStateLabel;
    const char* fromServerStateEnumToString(ServerState::E);
    ComputeDeviceInformationWidget* m_setComputeDeviceWidget;
    SetServerSettingsWidget* m_serverSettingsWidget;
    WaitingForConnectionWidget* m_waitingForConnectionWidget;
    ReadyForRenderingWidget* m_readyForRenderingWidget;

    // SERVER
    ServerState::E m_serverState;
    RenderServerState::E m_renderState;
    QTcpSocket *m_clientSocket;
    QTcpServer *m_server;
    void setServerState(ServerState::E state);
    ushort m_serverPort;
    ComputeDevice* m_computeDevice;
    RenderServer& m_renderServer;
    ComputeDeviceRepository m_computeDeviceRepository;

    void onStateSetComputeDeviceEnter();
    void onStateSetComputeDeviceExit();
    void onStateSetServerSettingsEnter();
    void onStateSetServerSettingsExit();
    void onStateWaitForConnectionEnter();
    void onStateWaitForConnectionExit();
    void onStateReadyForRenderingEnter();
    void onStateReadyForRenderingExit();
    void resetProcess();

};

#endif // MAINWINDOW_H
