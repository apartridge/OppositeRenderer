/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include <QObject>
#include "gui_export_api.h"

class OutputSettingsModel : public QObject
{
    Q_OBJECT;

public:
    GUI_EXPORT_API OutputSettingsModel(void);
    GUI_EXPORT_API ~OutputSettingsModel(void);
    GUI_EXPORT_API void setWidth(unsigned int width);
    GUI_EXPORT_API void setHeight(unsigned int height);
    GUI_EXPORT_API unsigned int getWidth() const;
    GUI_EXPORT_API unsigned int getHeight() const;
    GUI_EXPORT_API float getGamma() const;
    GUI_EXPORT_API float getAspectRatio() const;
    GUI_EXPORT_API void setGamma(float gamma);

signals:
    void resolutionUpdated();
    void gammaUpdated();

private:
    unsigned int m_width;
    unsigned int m_height;
    float m_gamma;
};

