/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "ConnectedServersTableModel.hxx"
#include "../client/RenderServerConnections.hxx"
#include "../client/RenderServerConnection.hxx"

ConnectedServersTableModel::ConnectedServersTableModel(QObject* parent, const RenderServerConnections & serverConnections ) :
    QAbstractTableModel(parent),
    m_serverConnections(serverConnections)
{
    connect(&m_serverConnections, SIGNAL(serversStateUpdated()), this, SLOT(onServersInfoUpdated()));
}

QVariant ConnectedServersTableModel::data( const QModelIndex &index, int role ) const
{
    if(role == Qt::DisplayRole)
    {
        const RenderServerConnection & connection = m_serverConnections.at(index.row());
        switch(index.column())
        {
        case 0:
            return connection.getComputeDeviceName();
        case 1:
            return QString("%1:%2").arg(connection.getServerIp(), connection.getServerPort());
        case 2:
            return QString("%1 sec").arg(connection.getRenderTimeSeconds(), 0, 'f', 1);
        case 3:
            return QString("%1 sec").arg(connection.getTotalTimeSeconds(), 0, 'f', 1);
        case 4:
            return QString::number(connection.getNumIterationsReceived());
        case 5:
            return QString("%1/sec").arg(connection.getNumIterationsReceived()/connection.getTotalTimeSeconds(), 0, 'f', 2);
        case 6:
            return QString("%1/%2").arg(connection.getNumPendingIterations()).arg(connection.getPendingIterationsLimit());
        case 7:
            return QString("%1 %").arg(connection.getServerEfficiency()*100, 0, 'f', 1);
        case 8:
            return QString("%1 MB").arg( float(connection.getBytesReceived())/float(1024*1024), 0, 'f', 1);
        case 9:
            return connection.getTotalTimeSeconds() > 0 ?
                  QString("%1 MB/s").arg( float(connection.getBytesReceived())/float(1024*1024) / connection.getTotalTimeSeconds(), 0, 'f', 1)
                : "-";
        case 10:
            return
                QString("%1, %2/sec")
                .arg(connection.getNumPacketsReceived())
                .arg(connection.getTotalTimeSeconds() > 0 ? connection.getNumPacketsReceived()/float(connection.getTotalTimeSeconds()) : 0, 0, 'f', 1);
        case 11:
            return QString("%1 sec").arg(connection.getAverageRequestResponseTime(), 0, 'f', 2);
        case 12:
            return connection.getNumPacketsReceived() > 0 ?
                QString("%1 (max %2)")
                     .arg(connection.getNumIterationsReceived()/float(connection.getNumPacketsReceived()), 0, 'f', 1)
                     .arg(connection.getMaxIterationsPerPacket())
                : "-";
        case 13:
            return connection.getNumPacketsReceived() > 0 ?
                QString("%1 MB/packet").arg(connection.getBytesReceived()/float(1024*1024)/float(connection.getNumPacketsReceived()), 0, 'f', 1)
                : "-";
        case 14:
            return renderServerStateEnumToString(connection.getRenderServerState());
        }
    }

    return QVariant();
}

int ConnectedServersTableModel::columnCount( const QModelIndex &parent ) const
{
    return 15;
}

int ConnectedServersTableModel::rowCount( const QModelIndex & parent ) const
{
    return m_serverConnections.numServers();
}

QModelIndex ConnectedServersTableModel::index( int row, int column, const QModelIndex & parent) const
{
    return createIndex(row, column, nullptr);
}

QVariant ConnectedServersTableModel::headerData( int section, Qt::Orientation orientation, int role) const
{
    if(role != Qt::DisplayRole)
    {
        return QVariant();
    }

    switch(section)
    {
    case 0: return "Compute Device";
    case 1: return "IP-address";
    case 2: return "Render Time";
    case 3: return "Total Time";
    case 4: return "Iterations";
    case 5: return "Iterations/sec";
    case 6: return "Pending ";
    case 7: return "Render %";
    case 8: return "Received data";
    case 9: return "MB/second";
    case 10: return "Packets";
    case 11: return "Avg Req-Resp time";
    case 12: return "Iterations/packet";
    case 13: return "MB/packet";
    case 14: return "State";
    }
    return "";
}

void ConnectedServersTableModel::onServersInfoUpdated()
{
    beginResetModel();
    endResetModel();
}