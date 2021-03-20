/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#ifndef SceneDock_H
#define SceneDock_H

#include <QDockWidget>

namespace Ui {
class SceneDock;
}

class SceneManager;

class SceneDock : public QDockWidget
{
    Q_OBJECT
    
public:
    explicit SceneDock(QWidget *parent, SceneManager & sceneManager);
    ~SceneDock();

public slots:
    void onSceneUpdated();
    void onSceneLoadingNew();

private:
    Ui::SceneDock *ui;
    SceneManager & m_sceneManager;
};

#endif // SceneDock_H
