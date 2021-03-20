/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "RenderServerConnections.hxx"
#include "RenderServerConnection.hxx"
#include <QThread>

RenderServerConnections::RenderServerConnections()
{

}

RenderServerConnections::~RenderServerConnections()
{
    for(int i = 0; i < m_serverConnections.size(); i++)
    {
        QMetaObject::invokeMethod(m_serverConnections.at(i), "onAboutToQuit", Qt::QueuedConnection);
        QMetaObject::invokeMethod(m_serverConnectionThreads[i], "quit", Qt::QueuedConnection);
        m_serverConnectionThreads[i]->quit();
    }


    for(int i = 0; i < m_serverConnections.size(); i++)
    {
        m_serverConnectionThreads[i]->wait();
    }
}

void RenderServerConnections::push_back( RenderServerConnection* connection )
{
    //printf("RenderServerConnections thread ptr: %u \n", (unsigned int)this->thread());
    //printf("connection thread ptr: %u \n", (unsigned int)connection->thread());

    // Add the connection to our set and fire it up in a new thread
    m_serverConnections.push_back(connection);
    m_serverConnectionThreads.push_back(new QThread());
    m_serverConnections.back()->moveSocketToThread(m_serverConnectionThreads.back());
    m_serverConnections.back()->moveToThread(m_serverConnectionThreads.back());
    connect(m_serverConnectionThreads.back(), SIGNAL(started()), m_serverConnections.back(), SLOT(onThreadStarted()));
    m_serverConnectionThreads.back()->start();

    connect(m_serverConnections.back(), SIGNAL(newRenderServerState(RenderServerState::E)), 
        this, SIGNAL(serversStateUpdated()), Qt::QueuedConnection);
    connect(m_serverConnections.back(), SIGNAL(stateUpdated()), 
        this, SIGNAL(serversStateUpdated()), Qt::QueuedConnection);

    emit serverConnectionAdded();
    emit serversStateUpdated();
}

const RenderServerConnection & RenderServerConnections::at( int index ) const
{
    return *m_serverConnections.at(index);
}

RenderServerConnection & RenderServerConnections::at( int index )
{
    return *m_serverConnections.at(index);
}

int RenderServerConnections::numRenderingServers() const
{
    int sum = 0;
    for(int i = 0; i < m_serverConnections.size(); i++)
    {
        if(m_serverConnections.at(i)->getRenderServerState() == RenderServerState::RENDERING)
        {
            sum++;
        }
    }
    return sum;
}

int RenderServerConnections::numServers() const
{
    return static_cast<int>(m_serverConnections.size());
}
