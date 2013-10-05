/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

/*
Note that there are two RenderServerState enums, one for Server and one for Client
*/
#pragma once
namespace RenderServerState
{
    enum E {NOT_VALID_RENDER_STATE,
            WAITING_FOR_INTRODUCTION_REQUEST,
            READY,
            RENDERING,
            ERROR_UNKNOWN};
}
const char* renderStateEnumToText(RenderServerState::E state);