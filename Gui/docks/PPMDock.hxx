/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#ifndef PPMDock_H
#define PPMDock_H

#include <QDockWidget>

namespace Ui {
class PPMDock;
}
class Application;
class PPMSettingsModel;

class PPMDock : public QDockWidget
{
    Q_OBJECT
    
public:
    explicit PPMDock(QWidget *parent, const Application & application, PPMSettingsModel & PPMSettings);
    ~PPMDock();

public slots:
    void onRenderStatisticsUpdated();

private slots:
    void onFormSubmitted();
    void onModelUpdated();

private:
    Ui::PPMDock *ui;
    const Application & m_application;
    PPMSettingsModel & m_PPMSettingsModel;
};

#endif // PPMDock_H
