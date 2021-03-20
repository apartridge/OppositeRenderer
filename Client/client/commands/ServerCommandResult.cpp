/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#include "ServerCommandResult.h"

ServerCommandResult::ServerCommandResult()
    : m_success(false)
    , m_newRenderServerState(RenderServerState::ERROR_UNKNOWN)
{
}

ServerCommandResult::ServerCommandResult(bool success, RenderServerState state)
    : m_success(success)
    , m_newRenderServerState(state)
{
}

RenderServerState ServerCommandResult::getNewRenderServerState() const
{
    return m_newRenderServerState;
}

void ServerCommandResult::setNewRenderServerState(RenderServerState val)
{
    m_newRenderServerState = val;
}

bool ServerCommandResult::getSuccess() const
{
    return m_success;
}

void ServerCommandResult::setSuccess(bool val)
{
    m_success = val;
}
