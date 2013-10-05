/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "ServerCommand.h"

class GetServerDetailsCommand : public ServerCommand
{
public:
    GetServerDetailsCommand();
    virtual ~GetServerDetailsCommand(void);
    virtual void executeCommand( QTcpSocket & socket);
    virtual ServerCommandResult onResponseReady(RenderServerConnection & , QTcpSocket & socket);
    virtual RenderServerState::E getInitialRenderServerState() const;
};

