/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once

#include <QAbstractTableModel>
class RenderServerConnections;

class ConnectedServersTableModel : public QAbstractTableModel
{
    Q_OBJECT;
public:
    ConnectedServersTableModel(QObject* parent, const RenderServerConnections & serverConnections );
    virtual int rowCount ( const QModelIndex & parent = QModelIndex() ) const;
    virtual int columnCount( const QModelIndex &parent = QModelIndex( ) ) const;
    virtual QVariant data( const QModelIndex &index, int role = Qt::DisplayRole ) const;
    virtual QModelIndex index ( int row, int column, const QModelIndex & parent = QModelIndex() ) const;
    virtual QVariant headerData( int section, Qt::Orientation orientation, int role = Qt::DisplayRole ) const;

private slots:
    void onServersInfoUpdated();

private:
    const RenderServerConnections & m_serverConnections;
    void abCde(int abc);
};