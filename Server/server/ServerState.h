/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#pragma once
enum class ServerState
{
    SET_COMPUTE_DEVICE,
    SET_SERVER_SETTINGS,
    WAIT_FOR_CLIENT_CONNECTION,
    READY_FOR_RENDERING
};

const char* serverStateEnumToString(ServerState state);