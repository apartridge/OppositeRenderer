/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#pragma once
#include "../RenderServerState.h"

class ServerCommandResult
{
public:
    ServerCommandResult();
    ServerCommandResult(bool success, RenderServerState state = RenderServerState::ERROR_UNKNOWN);
    RenderServerState getNewRenderServerState() const;
    void setNewRenderServerState(RenderServerState val);
    bool getSuccess() const;
    void setSuccess(bool val);

private:
    bool m_success;
    RenderServerState m_newRenderServerState;
};
