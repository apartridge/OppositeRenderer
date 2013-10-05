/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "GetServerDetailsCommand.h"
#include "client/RenderServerConnection.hxx"
#include <QTcpSocket>
 
GetServerDetailsCommand::GetServerDetailsCommand()
{

}

GetServerDetailsCommand::~GetServerDetailsCommand(void)
{

}

void GetServerDetailsCommand::executeCommand( QTcpSocket & socket )
{
    char command[256];
    sprintf(command, "GET SERVER DETAILS\n");
    QByteArray a(command);
    socket.write(a);
}

ServerCommandResult GetServerDetailsCommand::onResponseReady(RenderServerConnection & connection, QTcpSocket & socket )
{
    QDataStream stream(&socket);
    QString computeDeviceName;
    stream >> computeDeviceName;

    if(stream.status() != QDataStream::Ok)
    {
        printf("In GetServerDetailsCommand::onResponseReady, stream.status() != QDataStream::Ok\n");
    }

    if(computeDeviceName.length() < 4 || computeDeviceName.length() > 255)
    {
        return ServerCommandResult(false, RenderServerState::ERROR_INVALID_CONFIRMATION);
    }

    connection.setComputeDeviceName(computeDeviceName);

    return ServerCommandResult(true, RenderServerState::RENDERING);
}

RenderServerState::E GetServerDetailsCommand::getInitialRenderServerState() const
{
    return RenderServerState::WAITING_FOR_DEVICE_INFORMATION;
}
