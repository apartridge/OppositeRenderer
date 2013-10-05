/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include <QObject>
#include "gui_export_api.h"
class PPMSettingsModel : public QObject
{
    Q_OBJECT;

public:
    GUI_EXPORT_API PPMSettingsModel(void);
    GUI_EXPORT_API ~PPMSettingsModel(void);
    GUI_EXPORT_API double getPPMInitialRadius() const;
    GUI_EXPORT_API void setPPMInitialRadius(double PPMInitialRadius);

signals:
    void updated();

private:
    double m_PPMInitialRadius;
};

