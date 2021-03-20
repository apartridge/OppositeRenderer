/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#pragma once
#include "../gui_export_api.h"

#include <memory>

class IScene;
class SceneFactory
{
public:
    GUI_EXPORT_API std::unique_ptr<IScene> getSceneByName(const char* name);
};