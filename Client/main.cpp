/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#include "DistributedApplication.h"
#include "MainWindowBase.h"
#include "client/RenderServerState.h"
#include "gui/ClientMainWindow.h"

#include <QApplication>
#include <QThread>

int main(int argc, char** argv)
{
    qRegisterMetaType<RenderServerState>("RenderServerState");
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
    int returnCode = qApplication.exec();

    QMetaObject::invokeMethod(&application, "onAboutToQuit", Qt::QueuedConnection);
    application.wait();

    return returnCode;
}
