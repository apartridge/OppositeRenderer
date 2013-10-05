/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#ifndef RENDER_ENGINE_EXPORT_API 
#   ifdef RENDER_ENGINE_DLL
#       define RENDER_ENGINE_EXPORT_API __declspec(dllexport)
#       define RENDER_ENGINE_EXPORT_API_QT Q_DECL_EXPORT
#   else
#       define RENDER_ENGINE_EXPORT_API __declspec(dllimport)
#       define RENDER_ENGINE_EXPORT_API_QT Q_DECL_IMPORT
#   endif
#endif