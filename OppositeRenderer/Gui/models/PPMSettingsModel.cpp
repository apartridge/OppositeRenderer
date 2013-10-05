/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "PPMSettingsModel.hxx"

PPMSettingsModel::PPMSettingsModel(void)
    : m_PPMInitialRadius(0.0f)
{
}

PPMSettingsModel::~PPMSettingsModel(void)
{
}

double PPMSettingsModel::getPPMInitialRadius() const
{
    return m_PPMInitialRadius;
}

void PPMSettingsModel::setPPMInitialRadius( double PPMInitialRadius )
{
    m_PPMInitialRadius = PPMInitialRadius;
    emit updated();
}
