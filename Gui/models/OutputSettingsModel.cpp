/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#include "OutputSettingsModel.h"

OutputSettingsModel::OutputSettingsModel()
    : m_width(0)
    , m_height(0)
    , m_gamma(2.2)
{
}

void OutputSettingsModel::setWidth(unsigned int width)
{
    const bool shouldEmit = (m_width != width);
    m_width = width;
    if (shouldEmit)
    {
        emit resolutionUpdated();
    }
}

void OutputSettingsModel::setHeight(unsigned int height)
{
    const bool shouldEmit = (m_height != height);
    m_height = height;
    if (shouldEmit)
    {
        emit resolutionUpdated();
    }
}

unsigned int OutputSettingsModel::getWidth() const
{
    return m_width;
}

unsigned int OutputSettingsModel::getHeight() const
{
    return m_height;
}

float OutputSettingsModel::getGamma() const
{
    return m_gamma;
}

void OutputSettingsModel::setGamma(float gamma)
{
    const bool shouldEmit = (m_gamma != gamma);
    m_gamma = gamma;
    if (shouldEmit)
    {
        emit gammaUpdated();
    }
}

float OutputSettingsModel::getAspectRatio() const
{
    return float(m_width) / float(m_height);
}
