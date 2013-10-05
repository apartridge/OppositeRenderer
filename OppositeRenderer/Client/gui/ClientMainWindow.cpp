/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "ClientMainWindow.hxx"
#include "DistributedApplication.hxx"
#include <QAction>
#include "dialogs/AddNewServerConnectionDialog.hxx"
#include "docks/ConnectedServersDock.hxx"
#include "docks/RenderServersSummaryDock.hxx"

ClientMainWindow::ClientMainWindow(DistributedApplication & application) 
    : m_application(application),
      m_mainWindowBase(MainWindowBase(application))
{
    QAction* aa = new QAction(this);
    aa->setObjectName(QString::fromUtf8("aa"));
    aa->setText("About Server Manager");

    // Add new render server action

    QAction* actionConnectToNewRenderServer = new QAction(this);
    actionConnectToNewRenderServer->setObjectName(QString::fromUtf8("aa"));
    actionConnectToNewRenderServer->setText("Add new render server");
    m_mainWindowBase.menuFile->insertAction(m_mainWindowBase.actionOpen_scene, actionConnectToNewRenderServer);
    m_mainWindowBase.setWindowTitle("Opposite Renderer");
    connect(actionConnectToNewRenderServer, SIGNAL(triggered()), this, SLOT(onActionConnectToNewRenderServer()));

    // Add connected servers dock

    ConnectedServersDock* dock = new ConnectedServersDock(&m_mainWindowBase, application.getServerConnections());
    m_mainWindowBase.addDockWidget(Qt::BottomDockWidgetArea, dock);

    // Add render server summary dock

    RenderServersSummaryDock* dock2 = new RenderServersSummaryDock(&m_mainWindowBase, application);
    m_mainWindowBase.addDockWidget(Qt::RightDockWidgetArea, dock2);
    connect(dock2, SIGNAL(actionConnectToNewRenderServer()), actionConnectToNewRenderServer, SIGNAL(triggered()));

    connect(this, SIGNAL(hasNewServerConnectionSocket(QTcpSocket*)), 
        &m_application, SLOT(onNewServerConnectionSocket(QTcpSocket*)));

    onActionConnectToNewRenderServer();
}

void ClientMainWindow::show()
{
    return m_mainWindowBase.showMaximized();
}

void ClientMainWindow::onActionConnectToNewRenderServer()
{
    AddNewServerConnectionDialog* dialog = new AddNewServerConnectionDialog(&m_mainWindowBase, m_application.thread());
    connect(dialog, SIGNAL(hasNewServerConnectionSocket(QTcpSocket*)), 
            this, SLOT(onNewServerConnectionSocket(QTcpSocket*)), Qt::DirectConnection);
    dialog->open();
}

void ClientMainWindow::onNewServerConnectionSocket(QTcpSocket* socket)
{
    // Pass ownership of socket to the recipient
    socket->moveToThread(m_application.thread());
    emit hasNewServerConnectionSocket(socket);
}