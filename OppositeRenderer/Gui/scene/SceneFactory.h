/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "gui_export_api.h"
class IScene;
class SceneFactory
{
public:
    GUI_EXPORT_API SceneFactory(void);
    GUI_EXPORT_API ~SceneFactory(void);
    GUI_EXPORT_API IScene* getSceneByName(const char* name);
};