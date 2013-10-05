/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#ifndef CameraDock_H
#define CameraDock_H

#include <QDockWidget>

namespace Ui {
class CameraDock;
}

class Camera;
class PPMSettingsModel;
class OutputSettingsModel;

class CameraDock : public QDockWidget
{
    Q_OBJECT
    
public:
    explicit CameraDock(QWidget *parent, Camera & camera,  PPMSettingsModel & ppmModel,  OutputSettingsModel & outputModel);
    ~CameraDock();

public slots:
    void onCameraUpdated();

signals:
    void cameraUpdated();

private slots:
    void onUpdate();
    void onCornell();
    void onSponza();
    void onConference();

private:
    Ui::CameraDock *ui;
    Camera & m_camera;
    PPMSettingsModel & m_PPMModel;
    OutputSettingsModel & m_outputModel;
};

#endif // CameraDock_H
