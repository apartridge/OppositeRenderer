/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once

/*
Note that there are two RenderServerState enums, one for Server and one for Client
*/

namespace RenderServerState
{
    enum E
    {
        NO_DEVICE_INFORMATION,
        WAITING_FOR_DEVICE_INFORMATION,
        RENDERING,
        SENDING_RENDER_COMMAND,
        DISCONNECTED,
        ERROR_SOCKET,
        ERROR_INVALID_CONFIRMATION,
        ERROR_COMMAND_TIMEOUT,
        ERROR_UNKNOWN
    };
}

const char* renderServerStateEnumToString(RenderServerState::E state);