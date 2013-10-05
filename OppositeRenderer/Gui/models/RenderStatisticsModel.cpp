/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "RenderStatisticsModel.hxx"

RenderStatisticsModel::RenderStatisticsModel(void)
    : m_currentPPMRadius(0.0f),
      m_numIterations(0),
      m_numPreviewedIterations(0),
      m_numEmittedPhotons(0),
      m_numEmittedPhotonsPerIteration(0)
{
}

RenderStatisticsModel::~RenderStatisticsModel(void)
{
}

unsigned long long RenderStatisticsModel::getNumIterations() const
{
    return m_numIterations;
}

void RenderStatisticsModel::setNumIterations( unsigned long long numIterations )
{
    m_numIterations = numIterations;
    emit updated();
}

unsigned long long RenderStatisticsModel::getNumPhotonsInEstimate() const
{
    return m_numPhotonsInEstimate;
}

void RenderStatisticsModel::setNumPhotonsInEstimate( unsigned long long numberOfPhotonsInEstimate )
{
    m_numPhotonsInEstimate = numberOfPhotonsInEstimate;
}

unsigned long long RenderStatisticsModel::getNumEmittedPhotons() const
{
    return m_numEmittedPhotons;
}

void RenderStatisticsModel::setNumEmittedPhotons( unsigned long long numberOfEmittedPhotons )
{
    m_numEmittedPhotons = numberOfEmittedPhotons;
}

unsigned long long RenderStatisticsModel::getNumEmittedPhotonsPerIteration() const
{
    return m_numEmittedPhotonsPerIteration;
}

void RenderStatisticsModel::setNumEmittedPhotonsPerIteration( unsigned long long numberOfEmittedPhotonsPerIteration )
{
    m_numEmittedPhotonsPerIteration = numberOfEmittedPhotonsPerIteration;
}

double RenderStatisticsModel::getCurrentPPMRadius() const
{
    return m_currentPPMRadius;
}

void RenderStatisticsModel::setCurrentPPMRadius( double currentPPMRadius )
{
    m_currentPPMRadius = currentPPMRadius;
}

unsigned long long RenderStatisticsModel::getNumPreviewedIterations() const
{
    return m_numPreviewedIterations;
}

void RenderStatisticsModel::setNumPreviewedIterations( unsigned long long numPreviewedIterations )
{
    m_numPreviewedIterations = numPreviewedIterations;
}

void RenderStatisticsModel::incrementNumPreviewedIterations()
{
    m_numPreviewedIterations++;
}
