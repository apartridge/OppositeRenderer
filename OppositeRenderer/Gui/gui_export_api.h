/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#ifndef GUI_EXPORT_API
#   ifdef RENDERER_GUI_DLL
#       define GUI_EXPORT_API __declspec(dllexport)
#       define GUI_EXPORT_API_QT Q_DECL_EXPORT
#   else
#       define GUI_EXPORT_API __declspec(dllimport)
#       define GUI_EXPORT_API_QT Q_DECL_IMPORT
#   endif
#endif