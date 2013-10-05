/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "client/RenderServerState.h"

class ServerCommandResult
{
public:
    ServerCommandResult(void);
    ServerCommandResult(bool success, RenderServerState::E state = RenderServerState::ERROR_UNKNOWN);
    ~ServerCommandResult(void);
    RenderServerState::E getNewRenderServerState() const;
    void setNewRenderServerState(RenderServerState::E val);
    bool getSuccess() const;
    void setSuccess(bool val);

private:
    bool m_success;
    RenderServerState::E m_newRenderServerState;
};

