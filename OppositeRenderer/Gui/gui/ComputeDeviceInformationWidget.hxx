/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#ifndef GPU_DEVICE_INFORMATION_DIALOG_H
#define GPU_DEVICE_INFORMATION_DIALOG_H

#include <QWidget>
#include "ui/ui_ComputeDeviceInformationWidget.h"
#include <vector>
#include "ComputeDeviceInformationWidgetTabPage.hxx"
#include "gui_export_api.h"

class ComputeDevice;
class ComputeDeviceRepository;

class ComputeDeviceInformationWidget : public QWidget, public Ui::ComputeDeviceInformationWidget
{
    Q_OBJECT

public:
    GUI_EXPORT_API ComputeDeviceInformationWidget(QWidget* parent, ComputeDeviceRepository & computeDeviceRepo);

signals:
    GUI_EXPORT_API_QT void hasSelectedComputeDevice(ComputeDevice* selectedComputeDevice);

private slots:
    void onHasSelectedComputeDevice(ComputeDevice*);

private:
    void setupGpuDeviceTabPages(ComputeDeviceRepository & computeDeviceRepo);
    std::vector<ComputeDeviceInformationWidgetTabPage*> m_tabPages;
};

#endif