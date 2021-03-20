/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#include "SceneFactory.h"
#include "scene/Cornell.h"
#include "scene/IScene.h"

std::unique_ptr<IScene> SceneFactory::getSceneByName(const char* name)
{
    if (strcmp(name, "Cornell") == 0)
    {
        return std::make_unique<Cornell>();
    }
    else
    {
        return Scene::createFromFile(name);
    }
}
