/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#pragma once
#include "../client/RenderServerConnection.hxx"
#include "MainWindowBase.hxx"

#include <QObject>

class DistributedApplication;

class ClientMainWindow : public QObject
{
    Q_OBJECT;

public:
    ClientMainWindow(DistributedApplication& application);
    void show();

signals:
    void hasNewServerConnectionSocket(QTcpSocket*);

private slots:
    void onActionConnectToNewRenderServer();
    void onNewServerConnectionSocket(QTcpSocket*);

private:
    MainWindowBase m_mainWindowBase;
    DistributedApplication& m_application;
};