/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#ifndef WAITINGFORCONNECTIONWIDGET_H
#define WAITINGFORCONNECTIONWIDGET_H

#include <QWidget>

namespace Ui {
class WaitingForConnectionWidget;
}

class WaitingForConnectionWidget : public QWidget
{
    Q_OBJECT
    
public:
    explicit WaitingForConnectionWidget(QWidget *parent = 0);
    ~WaitingForConnectionWidget();
    void setPortNumber(QString& string);
    
private:
    Ui::WaitingForConnectionWidget *ui;
};

#endif // WAITINGFORCONNECTIONWIDGET_H
