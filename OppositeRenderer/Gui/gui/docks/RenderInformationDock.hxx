/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#ifndef RENDERINFORMATIONDOCK_H
#define RENDERINFORMATIONDOCK_H

#include <QDockWidget>
#include "RunningStatus.h"

namespace Ui {
class RenderInformationDock;
}

class Application;
class RenderStatisticsModel;

class RenderInformationDock : public QDockWidget
{
    Q_OBJECT
    
public:
    explicit RenderInformationDock(QWidget *parent, const RenderStatisticsModel & renderStatisticsModel, Application & application);
    ~RenderInformationDock();

signals:
    void renderStatusToggle();
    void renderRestart();

private slots:
    void onRunningStatusChanged();
    void onRenderStatisticsUpdated();
    void onUpdateRenderTime();
    
private:
    Ui::RenderInformationDock *ui;
    Application & m_application;
    const RenderStatisticsModel & m_renderStatisticsModel;
};

#endif // RENDERINFORMATIONDOCK_H
