/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once

#include "IScene.h"
#include "renderer/Light.h"
#include "render_engine_export_api.h"
#include <QVector>
#include <QByteArray>
#include "math/AAB.h"

struct aiScene;
struct aiMesh;
class Material;
struct aiNode;
class DiffuseEmitter;
struct aiColor3D;
class QFileInfo;

namespace Assimp
{
    class Importer;
}

class Scene : public IScene
{
public:
    Scene(void);
    RENDER_ENGINE_EXPORT_API virtual ~Scene(void);
    RENDER_ENGINE_EXPORT_API static IScene* createFromFile(const char* file);
    virtual optix::Group getSceneRootGroup(optix::Context & context);
    void loadDefaultSceneCamera();
    virtual const QVector<Light> & getSceneLights() const;
    virtual Camera getDefaultCamera() const;
    virtual const char* getSceneName() const;
    virtual AAB getSceneAABB() const ;
    RENDER_ENGINE_EXPORT_API virtual unsigned int getNumTriangles() const;

private:
    optix::Geometry Scene::createGeometryFromMesh(aiMesh* mesh, optix::Context & context);
    void loadMeshLightSource( aiMesh* mesh, DiffuseEmitter* diffuseEmitter );
    optix::Group getGroupFromNode(optix::Context & context, aiNode* node, QVector<optix::Geometry> & geometries, QVector<Material*> & materials);
    optix::GeometryInstance getGeometryInstance( optix::Context & context, optix::Geometry & geometry, Material* material );
    bool colorHasAnyComponent(const aiColor3D & color);
    void loadSceneMaterials();
    void loadLightSources();
    QVector<Material*> m_materials;
    QVector<Light> m_lights;
    QByteArray m_sceneName;
    QFileInfo* m_sceneFile; 
    Assimp::Importer* m_importer;
    const aiScene* m_scene;
    optix::Program m_intersectionProgram;
    optix::Program m_boundingBoxProgram;
    Camera m_defaultCamera;
    AAB m_sceneAABB;
    unsigned int m_numTriangles;

};