/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "Scene.h"
#include <cstdio>
#include <exception>

IScene::IScene()
{

}

IScene::~IScene()
{

}

// This base implementation finds a initial PPM radius by looking at the scene extent

float IScene::getSceneInitialPPMRadiusEstimate() const
{
    Vector3 sceneExtent = getSceneAABB().getExtent();
    float volume = sceneExtent.x*sceneExtent.y*sceneExtent.z;
    float cubelength = pow(volume, 1.f/3.f);
    float A = 6*cubelength*cubelength;
    float radius = A*3.94e-6;
    return radius;
}