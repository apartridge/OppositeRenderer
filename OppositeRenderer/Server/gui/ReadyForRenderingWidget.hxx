/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#ifndef READYFORRENDERINGWIDGET_H
#define READYFORRENDERINGWIDGET_H

#include <QWidget>

class QString;
class ComputeDevice;

namespace Ui {
class ReadyForRenderingWidget;
}

class ReadyForRenderingWidget : public QWidget
{
    Q_OBJECT
    
public:
    explicit ReadyForRenderingWidget(QWidget *parent = 0);
    ~ReadyForRenderingWidget();
    void appendToLog( const QString & );
    void clearLog();
    void setComputeDevice(const ComputeDevice & );
    void setClientName(const QString &);
    void setServerName(const QString &);
    void setRenderTime(float renderTime, float totalTime);
    void setPendingRenderCommandsAndIterations(unsigned int pendingRenderCommands, unsigned int pendingRenderIterations);
private:
    Ui::ReadyForRenderingWidget *ui;
};

#endif // READYFORRENDERINGWIDGET_H
