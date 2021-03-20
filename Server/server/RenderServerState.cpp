/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "RenderServerState.h"
const char* renderStateEnumToText(RenderServerState::E state)
{
    switch(state)
    {
    case RenderServerState::READY: return "Ready";
    case RenderServerState::RENDERING: return "Rendering";
    case RenderServerState::ERROR_UNKNOWN: return "Unknown error";
    case RenderServerState::WAITING_FOR_INTRODUCTION_REQUEST: return "Wait for introduction request.";
    }
    return "(Unknown state.)";
}