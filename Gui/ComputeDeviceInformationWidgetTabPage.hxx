/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#ifndef GPU_DEVICE_INFORMATION_DIALOG_TAB_PAGE_H
#define GPU_DEVICE_INFORMATION_DIALOG_TAB_PAGE_H

#include "ui/ui_ComputeDeviceInformationWidgetTabPage.h"

class ComputeDevice;

class ComputeDeviceInformationWidgetTabPage : public QWidget, private Ui::ComputeDeviceInformationWidgetTabPage
{
    Q_OBJECT

public:
    ComputeDeviceInformationWidgetTabPage(QWidget* parent, ComputeDevice & device);
    ComputeDevice& getComputeDevice();

signals:
    void hasSelectedComputeDevice(ComputeDevice* device);

private slots:
     void onButtonSelectComputeDevice();

private:
    ComputeDevice& m_device;
};

#endif