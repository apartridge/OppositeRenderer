#include "RenderServerState.h"

const char* renderServerStateEnumToString(RenderServerState::E state)
{
    switch(state)
    {
    case RenderServerState::NO_DEVICE_INFORMATION: return "No compute device information";
    case RenderServerState::WAITING_FOR_DEVICE_INFORMATION: return "Waiting for device information";
    case RenderServerState::RENDERING: return "OK/Ready";
    case RenderServerState::SENDING_RENDER_COMMAND: return "Sending render frame command to server";
    case RenderServerState::DISCONNECTED: return "Disconnected from server";
    case RenderServerState::ERROR_SOCKET: return "Error: Socket error";
    case RenderServerState::ERROR_UNKNOWN: return "Unknown error";
    case RenderServerState::ERROR_INVALID_CONFIRMATION: return "Error: Not valid confirmation from server on command";
    case RenderServerState::ERROR_COMMAND_TIMEOUT: return "Error: A command to server timed out / got no response";
    }
    return "(Unknown state)";
}
