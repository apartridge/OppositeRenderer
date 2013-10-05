/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "CameraDock.hxx"
#include "ui/ui_CameraDock.h"
#include <QMessageBox>
#include "renderer/Camera.h"
#include "models/PPMSettingsModel.hxx"
#include "models/OutputSettingsModel.hxx"

CameraDock::CameraDock(QWidget *parent, Camera & camera, PPMSettingsModel & ppmModel, OutputSettingsModel & outputModel) :
    QDockWidget(parent),
    ui(new Ui::CameraDock),
    m_camera(camera),
    m_PPMModel(ppmModel),
    m_outputModel(outputModel)
{
    ui->setupUi(this);
    this->setFeatures(QDockWidget::DockWidgetFloatable|QDockWidget::DockWidgetMovable);
    this->setAllowedAreas(Qt::LeftDockWidgetArea|Qt::RightDockWidgetArea);
    connect(ui->updateSettingsButton, SIGNAL(pressed()), this, SLOT(onUpdate()));
    connect(ui->cornellButton, SIGNAL(pressed()), this, SLOT(onCornell()));
    connect(ui->sponzaButton, SIGNAL(pressed()), this, SLOT(onSponza()));
    connect(ui->conferenceButton, SIGNAL(pressed()), this, SLOT(onConference()));


    onCameraUpdated();
}

CameraDock::~CameraDock()
{
    delete ui;
}

void CameraDock::onUpdate()
{
    bool ok[12];
    float cameraEyeX = ui->cameraEyeX->text().toFloat(&ok[0]);
    float cameraEyeY = ui->cameraEyeY->text().toFloat(&ok[1]);
    float cameraEyeZ = ui->cameraEyeZ->text().toFloat(&ok[2]);
    float cameraLookatX = ui->cameraLookatX->text().toFloat(&ok[3]);
    float cameraLookatY = ui->cameraLookatY->text().toFloat(&ok[4]);
    float cameraLookatZ = ui->cameraLookatZ->text().toFloat(&ok[5]);
    float cameraUpX = ui->cameraUpX->text().toFloat(&ok[6]);
    float cameraUpY = ui->cameraUpY->text().toFloat(&ok[7]);
    float cameraUpZ = ui->cameraUpZ->text().toFloat(&ok[8]);
    float cameraHfov = ui->cameraHfov->text().toFloat(&ok[9]);
    float cameraVfov = ui->cameraVfov->text().toFloat(&ok[10]);
    float cameraAperture = ui->cameraAperture->text().toFloat(&ok[11]);
    
    for(int i = 0; i < sizeof(ok)/sizeof(bool); i++)
    {
        if(!ok[i])
        {
            QMessageBox::information(this, "Invalid Camera Settings", "Please make sure that the values you inserted are valid/decimal values.");
            return;
        }
    }

    m_camera.eye.x = cameraEyeX;
    m_camera.eye.y = cameraEyeY;
    m_camera.eye.z = cameraEyeZ;
    m_camera.lookat.x = cameraLookatX;
    m_camera.lookat.y = cameraLookatY;
    m_camera.lookat.z = cameraLookatZ;
    m_camera.up.x = cameraUpX;
    m_camera.up.y = cameraUpY;
    m_camera.up.z = cameraUpZ;
    m_camera.hfov = cameraHfov;
    m_camera.vfov = cameraVfov;
    m_camera.aperture = cameraAperture;
    m_camera.setup();
    emit cameraUpdated();
}

static int precision(float x)
{
    if(fabs(x) < 1)
    {
        return 4;
    }
    else if(fabs(x) < 10)
    {
        return 3;
    }
    
    return 2;
}

void CameraDock::onCameraUpdated()
{
    ui->cameraEyeX->setText(QString::number(m_camera.eye.x, 'f', precision(m_camera.eye.x)));
    ui->cameraEyeY->setText(QString::number(m_camera.eye.y, 'f', precision(m_camera.eye.y)));
    ui->cameraEyeZ->setText(QString::number(m_camera.eye.z, 'f', precision(m_camera.eye.z)));

    ui->cameraLookatX->setText(QString::number(m_camera.lookat.x, 'f', precision(m_camera.lookat.x)));
    ui->cameraLookatY->setText(QString::number(m_camera.lookat.y, 'f', precision(m_camera.lookat.y)));
    ui->cameraLookatZ->setText(QString::number(m_camera.lookat.z, 'f', precision(m_camera.lookat.z)));

    ui->cameraUpX->setText(QString::number(m_camera.up.x, 'f',  precision(m_camera.up.x)));
    ui->cameraUpY->setText(QString::number(m_camera.up.y, 'f',  precision(m_camera.up.y)));
    ui->cameraUpZ->setText(QString::number(m_camera.up.z, 'f',  precision(m_camera.up.z)));

    ui->cameraHfov->setText(QString::number(m_camera.hfov, 'f', precision(m_camera.hfov)));
    ui->cameraVfov->setText(QString::number(m_camera.vfov, 'f', precision(m_camera.vfov)));
    ui->cameraAperture->setText(QString::number(m_camera.aperture, 'f', precision(m_camera.aperture)));
}


void CameraDock::onCornell()
{
    m_camera.eye.x = 0.4861;
    m_camera.eye.y = 2.192;
    m_camera.eye.z = 7.701;
    m_camera.lookat.x = 0.3500;
    m_camera.lookat.y = 2.179;
    m_camera.lookat.z = -1.50;
    m_camera.up.x = -0.0003;
    m_camera.up.y = 0.9999;
    m_camera.up.z = -0.0136;
    m_camera.hfov = 49.82;
    m_camera.vfov = 38.40;
    m_camera.aperture = 0.0150;
    m_camera.setup();
    m_PPMModel.setPPMInitialRadius(0.02);
    m_outputModel.setWidth(1024);
    m_outputModel.setHeight(768);
    emit cameraUpdated();
}

void CameraDock::onSponza()
{
    m_camera.eye.x = 6.852;
    m_camera.eye.y = 5.063;
    m_camera.eye.z = -0.50;
    m_camera.lookat.x = -6.70;
    m_camera.lookat.y = 3.10;
    m_camera.lookat.z = 0.80;
    m_camera.up.x = 0;
    m_camera.up.y = 1;
    m_camera.up.z = 0;
    m_camera.hfov = 54.87;
    m_camera.vfov = 32.56;
    m_camera.aperture = 0.0010;
    m_camera.setup();
    emit cameraUpdated();
    m_PPMModel.setPPMInitialRadius(0.08);
    m_outputModel.setWidth(1280);
    m_outputModel.setHeight(720);
}

void CameraDock::onConference()
{
    m_camera.eye.x = -7.70;
    m_camera.eye.y = 2.50;
    m_camera.eye.z = -3.30;
    m_camera.lookat.x = 9;
    m_camera.lookat.y = 0.80;
    m_camera.lookat.z = 4.30;
    m_camera.up.x = 0;
    m_camera.up.y = 1;
    m_camera.up.z = 0;
    m_camera.hfov = 49.82;
    m_camera.vfov = 29.28;
    m_camera.aperture = 0.0030;
    m_PPMModel.setPPMInitialRadius(0.02);
    m_camera.setup();
    emit cameraUpdated();
    m_outputModel.setWidth(1280);
    m_outputModel.setHeight(720);
}
