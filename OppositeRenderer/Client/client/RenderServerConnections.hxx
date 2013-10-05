/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include <vector>
#include <QObject>

class RenderServerConnection;
class QThread;

class RenderServerConnections : public QObject
{
    Q_OBJECT;
public:
    RenderServerConnections();
    ~RenderServerConnections();
    // Push back and pass ownership of the connection object
    void push_back(RenderServerConnection* connection);
    const RenderServerConnection & at(int index) const;
    RenderServerConnection & at(int index);
    int numServers() const;
    int numRenderingServers() const;

signals:
    void serverConnectionAdded();
    void serversStateUpdated();
private:
    std::vector<RenderServerConnection*> m_serverConnections;
    std::vector<QThread*> m_serverConnectionThreads;
    RenderServerConnections(const RenderServerConnections &);
    RenderServerConnections & operator=(const RenderServerConnections &);
};
