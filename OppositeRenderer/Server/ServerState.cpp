/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "ServerState.h"
const char* serverStateEnumToString(ServerState::E state)
{
    switch(state)
    {
    case ServerState::WAIT_FOR_CLIENT_CONNECTION: return "Awaiting connection from client.";
    case ServerState::SET_COMPUTE_DEVICE: return "Select compute device to use.";
    case ServerState::SET_SERVER_SETTINGS: return "Configure server settings.";
    case ServerState::READY_FOR_RENDERING: return "Ready for rendering.";
    }
    return "(Unknown state.)";
}