/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "ServerCommandResult.h"
#include "client/RenderServerState.h"

class QTcpSocket;
class RenderServerConnection;
class ServerCommand
{
public:
    ServerCommand(void);
    virtual ~ServerCommand(void) = 0;
    /*
    Execute a command and immediately return. Call onResponseReady() to wait for result.
    */
    virtual void executeCommand(QTcpSocket & m_socket) = 0;
    virtual ServerCommandResult onResponseReady(RenderServerConnection & connection, QTcpSocket & socket) = 0;
    virtual RenderServerState::E getInitialRenderServerState() const = 0;
};