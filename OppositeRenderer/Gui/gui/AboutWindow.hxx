/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#ifndef ABOUTWINDOW_H
#define ABOUTWINDOW_H

#include <QDialog>
#include "gui_export_api.h"

namespace Ui {
class AboutWindow;
}

class AboutWindow : public QDialog
{
    Q_OBJECT
    
public:
    explicit GUI_EXPORT_API AboutWindow(QWidget *parent = 0);
    GUI_EXPORT_API ~AboutWindow();
    
private:
    Ui::AboutWindow *ui;
};

#endif // ABOUTWINDOW_H
