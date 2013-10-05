/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include <QObject>
#include <QTime>
#include "gui_export_api.h"

class RenderStatisticsModel : public QObject
{
    Q_OBJECT;

public:
    GUI_EXPORT_API RenderStatisticsModel(void);
    GUI_EXPORT_API ~RenderStatisticsModel(void);
    GUI_EXPORT_API unsigned long long getNumIterations() const; 
    GUI_EXPORT_API void setNumIterations(unsigned long long numIterations); 

    GUI_EXPORT_API unsigned long long getNumPreviewedIterations() const; 
    GUI_EXPORT_API void setNumPreviewedIterations(unsigned long long numPreviewedIterations); 

    GUI_EXPORT_API unsigned long long  getNumPhotonsInEstimate() const;
    GUI_EXPORT_API void setNumPhotonsInEstimate(unsigned long long  numberOfPhotonsInEstimate); 
    GUI_EXPORT_API unsigned long long  getNumEmittedPhotons() const;
    GUI_EXPORT_API void setNumEmittedPhotons(unsigned long long  numberOfEmittedPhotons);
    GUI_EXPORT_API unsigned long long  getNumEmittedPhotonsPerIteration() const;
    GUI_EXPORT_API void setNumEmittedPhotonsPerIteration(unsigned long long  numberOfEmittedPhotonsPerIteration); 
    GUI_EXPORT_API double getCurrentPPMRadius() const;
    GUI_EXPORT_API void setCurrentPPMRadius(double currentPPMRadius); 
    GUI_EXPORT_API void incrementNumPreviewedIterations();

signals:
    void updated();

private:
    unsigned long long m_numEmittedPhotons;
    unsigned long long  m_numEmittedPhotonsPerIteration;
    double m_currentPPMRadius;
    unsigned long long m_numPhotonsInEstimate;
    unsigned long long m_numIterations;
    unsigned long long m_numPreviewedIterations;
};

