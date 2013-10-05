/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#ifndef SETSERVERSETTINGSWIDGET_H
#define SETSERVERSETTINGSWIDGET_H

#include <QWidget>

namespace Ui {
class SetServerSettingsWidget;
}

class SetServerSettingsWidget : public QWidget
{
    Q_OBJECT
    
public:
    explicit SetServerSettingsWidget(QWidget *parent = 0);
    ~SetServerSettingsWidget();
    ushort getPortNumber() const;

signals:
    void startServerFormSubmitted();

private slots:
    void onStartServerButtonClicked();

private:
    Ui::SetServerSettingsWidget *ui;
    ushort m_portNumber;
};

#endif // SETSERVERSETTINGSWIDGET_H
