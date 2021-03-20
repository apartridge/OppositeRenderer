/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#ifndef OutputDock_H
#define OutputDock_H

#include <QDockWidget>

namespace Ui {
class OutputDock;
}

class OutputSettingsModel;

class OutputDock : public QDockWidget
{
    Q_OBJECT
    
public:
    explicit OutputDock(QWidget *parent, OutputSettingsModel & model);
    ~OutputDock();

signals:
    void outputSettingsUpdated();

private slots:
    void onFormSubmitted();
    void onOutputSettingsModelUpdated();

private:
    Ui::OutputDock *ui;
    OutputSettingsModel & m_model;
};

#endif // OutputDock_H
