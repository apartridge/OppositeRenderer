/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "MainWindowBase.hxx"
#include "gui/ClientMainWindow.hxx"
#include "client/RenderServerState.h"
#include "DistributedApplication.hxx"

#include <QApplication>
#include <QThread>

int main( int argc, char** argv )
{
    qRegisterMetaType<RenderServerState::E>("RenderServerState::E");
    qRegisterMetaType<QAbstractSocket::SocketError>("QAbstractSocket::SocketError");

    QApplication qApplication(argc, argv);
    qApplication.setOrganizationName("Opposite Renderer");
    qApplication.setApplicationName("Opposite Renderer");

    DistributedApplication application(qApplication);
    ClientMainWindow mainWindow(application);

    QThread* m_thread = new QThread(&qApplication);
    application.moveToThread(m_thread);
    m_thread->start();

    mainWindow.show();
    int returnCode =  qApplication.exec();

    QMetaObject::invokeMethod(&application, "onAboutToQuit", Qt::QueuedConnection);
    application.wait();

    return returnCode;
}
