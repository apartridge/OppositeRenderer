/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#include "Scene.h"
#include "../config.h"
#include "../geometry_instance/AABInstance.h"
#include "../material/Diffuse.h"
#include "../material/DiffuseEmitter.h"
#include "../material/Glass.h"
#include "../material/Mirror.h"
#include "../material/ParticipatingMedium.h"
#include "../material/Texture.h"
#include "../util/ptxhelper.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QScopedPointer>
#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

Scene::Scene(void)
    : m_scene(NULL)
    , m_importer(new Assimp::Importer())
    , m_numTriangles(0)
    , m_sceneFile(NULL)
{
}

Scene::~Scene(void)
{
    printf("Delete scene\n");
    // deleting m_importer also deletes the scene
    delete m_importer;
    for (int i = 0; i < m_materials.size(); i++)
    {
        delete m_materials.at(i);
    }
    m_materials.clear();
    delete m_sceneFile;
}

static optix::float3 toFloat3(aiVector3D vector)
{
    return optix::make_float3(vector.x, vector.y, vector.z);
}

static optix::float3 toFloat3(aiColor3D vector)
{
    return optix::make_float3(vector.r, vector.g, vector.b);
}

static void minCoordinates(Vector3& min, const aiVector3D& vector)
{
    min.x = optix::fminf(min.x, vector.x);
    min.y = optix::fminf(min.y, vector.y);
    min.z = optix::fminf(min.z, vector.z);
}

static void maxCoordinates(Vector3& max, const aiVector3D& vector)
{
    max.x = optix::fmaxf(max.x, vector.x);
    max.y = optix::fmaxf(max.y, vector.y);
    max.z = optix::fmaxf(max.z, vector.z);
}

IScene* Scene::createFromFile(const char* filename)
{
    if (!QFile::exists(filename))
    {
        QString error = QString("The file that was supplied (%s) does not exist.").arg(filename);
        throw std::runtime_error(error.toStdString());
    }

    QScopedPointer<Scene> scenePtr(new Scene);
    scenePtr->m_sceneFile = new QFileInfo(filename);

    // Remove point and lines from the model
    scenePtr->m_importer->SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_POINT | aiPrimitiveType_LINE);

    scenePtr->m_scene = scenePtr->m_importer->ReadFile(
        filename,
        aiProcess_Triangulate | aiProcess_CalcTangentSpace | aiProcess_FindInvalidData | aiProcess_GenUVCoords
            | aiProcess_TransformUVCoords |
            // aiProcess_FindInstances          |
            aiProcess_JoinIdenticalVertices | aiProcess_OptimizeGraph | aiProcess_OptimizeMeshes |
            // aiProcess_PreTransformVertices   |
            aiProcess_GenSmoothNormals);

    if (!scenePtr->m_scene)
    {
        QString error = QString("An error occurred in Assimp during reading of this file: %1")
                            .arg(scenePtr->m_importer->GetErrorString());
        throw std::runtime_error(error.toStdString());
    }

    // Load materials

    scenePtr->loadSceneMaterials();

    // Load lights from file

    scenePtr->loadLightSources();

    // Find scene AABB and load any emitters

    Vector3 sceneAABBMin(1e33f);
    Vector3 sceneAABBMax(-1e33f);

    for (unsigned int i = 0; i < scenePtr->m_scene->mNumMeshes; i++)
    {
        aiMesh* mesh = scenePtr->m_scene->mMeshes[i];

        // Check if this is a diffuse emitter
        unsigned int materialIndex = mesh->mMaterialIndex;
        Material* geometryMaterial = scenePtr->m_materials.at(materialIndex);
        if (dynamic_cast<DiffuseEmitter*>(geometryMaterial) != NULL)
        {
            DiffuseEmitter* emitterMaterial = (DiffuseEmitter*)(geometryMaterial);
            scenePtr->loadMeshLightSource(mesh, emitterMaterial);
        }

        // Extend AABB

        scenePtr->m_numTriangles += mesh->mNumFaces;

        for (unsigned int j = 0; j < mesh->mNumFaces; j++)
        {
            aiFace face = mesh->mFaces[j];
            aiVector3D p1 = mesh->mVertices[face.mIndices[0]];
            aiVector3D p2 = mesh->mVertices[face.mIndices[0]];
            aiVector3D p3 = mesh->mVertices[face.mIndices[0]];
            minCoordinates(sceneAABBMin, p1);
            minCoordinates(sceneAABBMin, p2);
            minCoordinates(sceneAABBMin, p3);
            maxCoordinates(sceneAABBMax, p1);
            maxCoordinates(sceneAABBMax, p2);
            maxCoordinates(sceneAABBMax, p3);
        }
    }

    scenePtr->m_sceneAABB.min = sceneAABBMin;
    scenePtr->m_sceneAABB.max = sceneAABBMax;

    if (scenePtr->m_scene->mNumCameras > 0)
    {
        scenePtr->loadDefaultSceneCamera();
    }

    scenePtr->m_sceneName = QByteArray(scenePtr->m_sceneFile->absoluteFilePath().toLatin1().constData());
    return scenePtr.take();
}

void Scene::loadSceneMaterials()
{
    // printf("NUM MATERIALS: %d\n", m_scene->mNumMaterials);
    for (unsigned int i = 0; i < m_scene->mNumMaterials; i++)
    {
        aiMaterial* material = m_scene->mMaterials[i];
        aiString name;
        material->Get(AI_MATKEY_NAME, name);
        // printf("Material %d, %s:\n", i, name.C_Str());

        // Check if this is an Emitter
        aiColor3D emissivePower;
        if (material->Get(AI_MATKEY_COLOR_EMISSIVE, emissivePower) == AI_SUCCESS && colorHasAnyComponent(emissivePower))
        {
            aiColor3D diffuseColor;
            if (material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuseColor) != AI_SUCCESS)
            {
                diffuseColor.r = 1;
                diffuseColor.g = 1;
                diffuseColor.b = 1;
            }
            Material* material = new DiffuseEmitter(toFloat3(emissivePower), toFloat3(diffuseColor));
            m_materials.push_back(material);
            continue;
        }

        // Textured material
        aiString textureName;
        if (material->Get(AI_MATKEY_TEXTURE(aiTextureType_DIFFUSE, 0), textureName) == AI_SUCCESS)
        {
            QString textureAbsoluteFilePath
                = QString("%1/%2").arg(m_sceneFile->absoluteDir().absolutePath(), textureName.C_Str());

            // Use the displacement map as a normal map (in the crytek sponza test scene)
            aiString normalsName;
            Material* matl;
            if (material->Get(AI_MATKEY_TEXTURE(aiTextureType_NORMALS, 0), normalsName) == AI_SUCCESS)
            {
                printf("Found normal map %s!\n", normalsName.C_Str());
                QString normalsAbsoluteFilePath
                    = QString("%1/%2").arg(m_sceneFile->absoluteDir().absolutePath(), normalsName.C_Str());
                matl = new Texture(textureAbsoluteFilePath, normalsAbsoluteFilePath);
            }
            else
            {
                matl = new Texture(textureAbsoluteFilePath);
            }

            m_materials.push_back(matl);
            continue;
        }

        // Glass Material

        float indexOfRefraction;
        if (material->Get(AI_MATKEY_REFRACTI, indexOfRefraction) == AI_SUCCESS && indexOfRefraction > 1.0f)
        {
            // printf("\tGlass: IOR: %g\n", indexOfRefraction);
            Material* material = new Glass(indexOfRefraction, optix::make_float3(1, 1, 1));
            m_materials.push_back(material);
            continue;
        }

        // Reflective/mirror material
        aiColor3D reflectiveColor;
        if (material->Get(AI_MATKEY_COLOR_REFLECTIVE, reflectiveColor) == AI_SUCCESS
            && colorHasAnyComponent(reflectiveColor))
        {
            // printf("\tReflective color: %.2f %.2f %.2f\n", reflectiveColor.r, reflectiveColor.g, reflectiveColor.b);
            Material* material = new Mirror(toFloat3(reflectiveColor));
            m_materials.push_back(material);
            continue;
        }

        // Diffuse

        aiColor3D diffuseColor;
        if (material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuseColor) == AI_SUCCESS)
        {
            // printf("\tDiffuse %.2f %.2f %.2f\n", diffuseColor.r, diffuseColor.g, diffuseColor.b);
            Material* material = new Diffuse(toFloat3(diffuseColor));
            m_materials.push_back(material);
            continue;
        }

        // Fall back to a red diffuse material

        printf("\tError: Found no material instance to create for material index: %d\n", i);
        m_materials.push_back(new Diffuse(optix::make_float3(1, 0, 0)));
    }
}

void Scene::loadLightSources()
{
    for (unsigned int i = 0; i < m_scene->mNumLights; i++)
    {
        aiLight* lightPtr = m_scene->mLights[i];
        if (lightPtr->mType == aiLightSource_POINT)
        {
            Light light(toFloat3(lightPtr->mColorDiffuse), toFloat3(lightPtr->mPosition));
            m_lights.push_back(light);
        }
        else if (lightPtr->mType == aiLightSource_SPOT)
        {
            Light light(
                toFloat3(lightPtr->mColorDiffuse),
                toFloat3(lightPtr->mPosition),
                toFloat3(lightPtr->mDirection),
                lightPtr->mAngleInnerCone);
            m_lights.push_back(light);
        }
    }
}

void Scene::loadMeshLightSource(aiMesh* mesh, DiffuseEmitter* diffuseEmitter)
{
    // Convert mesh into a quad light source

    if (mesh->mNumFaces < 1 || mesh->mNumFaces > 2)
    {
        aiString name;
        m_scene->mMaterials[mesh->mMaterialIndex]->Get(AI_MATKEY_NAME, name);
        printf("Material %s: Does only support quadrangle light source NumFaces: %d.\n", name.C_Str(), mesh->mNumFaces);
    }

    aiFace face = mesh->mFaces[0];
    if (face.mNumIndices == 3)
    {
        optix::float3 anchor = toFloat3(mesh->mVertices[face.mIndices[0]]);
        optix::float3 p1 = toFloat3(mesh->mVertices[face.mIndices[1]]);
        optix::float3 p2 = toFloat3(mesh->mVertices[face.mIndices[2]]);
        optix::float3 v1 = p1 - anchor;
        optix::float3 v2 = p2 - anchor;

        Light light(diffuseEmitter->getPower(), anchor, v1, v2);
        m_lights.push_back(light);
        diffuseEmitter->setInverseArea(light.inverseArea);
    }
}

optix::Group Scene::getSceneRootGroup(optix::Context& context)
{
    if (!m_intersectionProgram)
    {
        std::string ptxFilename = getPtxFile("geometry_instance/TriangleMesh.ptx");
        m_intersectionProgram = context->createProgramFromPTXFile(ptxFilename, "mesh_intersect");
        m_boundingBoxProgram = context->createProgramFromPTXFile(ptxFilename, "mesh_bounds");
    }

    // printf("Sizeof materials array: %d", materials.size());

    // QVector<optix::GeometryInstance> instances;

    // Convert meshes into Geometry objects

    QVector<optix::Geometry> geometries;
    for (unsigned int i = 0; i < m_scene->mNumMeshes; i++)
    {
        optix::Geometry geometry = createGeometryFromMesh(m_scene->mMeshes[i], context);
        geometries.push_back(geometry);
        // optix::GeometryInstance instance = getGeometryInstanceFromMesh(m_scene->mMeshes[i], context, materials);
        // instances.push_back(instance);
    }

    // Convert nodes into a full scene Group

    optix::Group rootNodeGroup = getGroupFromNode(context, m_scene->mRootNode, geometries, m_materials);

#if ENABLE_PARTICIPATING_MEDIA
    {
        ParticipatingMedium partmedium = ParticipatingMedium(0.05, 0.01);
        AAB box = m_sceneAABB;
        box.addPadding(0.01);
        AABInstance participatingMediumCube(partmedium, box);
        optix::GeometryGroup group = context->createGeometryGroup();
        group->setChildCount(1);
        group->setChild(0, participatingMediumCube.getOptixGeometryInstance(context));
        optix::Acceleration a = context->createAcceleration("NoAccel", "NoAccel");
        group->setAcceleration(a);
        rootNodeGroup->setChildCount(rootNodeGroup->getChildCount() + 1);
        rootNodeGroup->setChild(rootNodeGroup->getChildCount() - 1, group);
    }
#endif

    optix::Acceleration acceleration = context->createAcceleration("Sbvh", "Bvh");
    rootNodeGroup->setAcceleration(acceleration);
    acceleration->markDirty();
    return rootNodeGroup;
}

optix::Geometry Scene::createGeometryFromMesh(aiMesh* mesh, optix::Context& context)
{
    unsigned int numFaces = mesh->mNumFaces;
    unsigned int numVertices = mesh->mNumVertices;

    optix::Geometry geometry = context->createGeometry();
    geometry->setPrimitiveCount(numFaces);
    geometry->setIntersectionProgram(m_intersectionProgram);
    geometry->setBoundingBoxProgram(m_boundingBoxProgram);

    // Create vertex, normal and texture buffer

    optix::Buffer vertexBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVertices);
    optix::float3* vertexBuffer_Host = static_cast<optix::float3*>(vertexBuffer->map());

    optix::Buffer normalBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVertices);
    optix::float3* normalBuffer_Host = static_cast<optix::float3*>(normalBuffer->map());

    geometry["vertexBuffer"]->setBuffer(vertexBuffer);
    geometry["normalBuffer"]->setBuffer(normalBuffer);

    // Copy vertex and normal buffers

    memcpy(
        static_cast<void*>(vertexBuffer_Host),
        static_cast<void*>(mesh->mVertices),
        sizeof(optix::float3) * numVertices);
    vertexBuffer->unmap();

    memcpy(
        static_cast<void*>(normalBuffer_Host), static_cast<void*>(mesh->mNormals), sizeof(optix::float3) * numVertices);
    normalBuffer->unmap();

    // Transfer texture coordinates to buffer
    optix::Buffer texCoordBuffer;
    if (mesh->HasTextureCoords(0))
    {
        texCoordBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, numVertices);
        optix::float2* texCoordBuffer_Host = static_cast<optix::float2*>(texCoordBuffer->map());
        for (unsigned int i = 0; i < mesh->mNumVertices; i++)
        {
            aiVector3D texCoord = (mesh->mTextureCoords[0])[i];
            texCoordBuffer_Host[i].x = texCoord.x;
            texCoordBuffer_Host[i].y = texCoord.y;
        }
        texCoordBuffer->unmap();
    }
    else
    {
        texCoordBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, 0);
    }

    geometry["texCoordBuffer"]->setBuffer(texCoordBuffer);

    // Tangents and bi-tangents buffers

    geometry["hasTangentsAndBitangents"]->setUint(mesh->HasTangentsAndBitangents() ? 1 : 0);
    if (mesh->HasTangentsAndBitangents())
    {
        optix::Buffer tangentBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVertices);
        optix::float3* tangentBuffer_Host = static_cast<optix::float3*>(tangentBuffer->map());
        memcpy(
            static_cast<void*>(tangentBuffer_Host),
            static_cast<void*>(mesh->mTangents),
            sizeof(optix::float3) * numVertices);
        tangentBuffer->unmap();

        optix::Buffer bitangentBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVertices);
        optix::float3* bitangentBuffer_Host = static_cast<optix::float3*>(bitangentBuffer->map());
        memcpy(
            static_cast<void*>(bitangentBuffer_Host),
            static_cast<void*>(mesh->mBitangents),
            sizeof(optix::float3) * numVertices);
        bitangentBuffer->unmap();

        geometry["tangentBuffer"]->setBuffer(tangentBuffer);
        geometry["bitangentBuffer"]->setBuffer(bitangentBuffer);
    }
    else
    {
        optix::Buffer emptyBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, 0);
        geometry["tangentBuffer"]->setBuffer(emptyBuffer);
        geometry["bitangentBuffer"]->setBuffer(emptyBuffer);
    }

    // Create index buffer

    optix::Buffer indexBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, numFaces);
    optix::int3* indexBuffer_Host = static_cast<optix::int3*>(indexBuffer->map());
    geometry["indexBuffer"]->setBuffer(indexBuffer);

    // Copy index buffer from host to device

    for (unsigned int i = 0; i < mesh->mNumFaces; i++)
    {
        aiFace face = mesh->mFaces[i];
        indexBuffer_Host[i].x = face.mIndices[0];
        indexBuffer_Host[i].y = face.mIndices[1];
        indexBuffer_Host[i].z = face.mIndices[2];
    }

    indexBuffer->unmap();

    return geometry;
}

void Scene::loadDefaultSceneCamera()
{
    aiNode* cameraNode = m_scene->mRootNode->FindNode(m_scene->mCameras[0]->mName);
    aiCamera* camera = m_scene->mCameras[0];

    aiVector3D eye = camera->mPosition;
    aiVector3D lookAt = eye + camera->mLookAt;
    aiVector3D up = camera->mUp;

    m_defaultCamera = Camera(
        Vector3(eye.x, eye.y, eye.z),
        Vector3(lookAt.x, lookAt.y, lookAt.z),
        Vector3(optix::normalize(optix::make_float3(up.x, up.y, up.z))),
        camera->mHorizontalFOV * 365.0f / (2.0f * M_PIf),
        camera->mHorizontalFOV * 365.0f / (2.0f * M_PIf),
        0.f,
        Camera::KeepHorizontal);
}

optix::Group Scene::getGroupFromNode(
    optix::Context& context, aiNode* node, QVector<optix::Geometry>& geometries, QVector<Material*>& materials)
{
    if (node->mNumMeshes > 0)
    {
        QVector<optix::GeometryInstance> instances;
        optix::GeometryGroup geometryGroup = context->createGeometryGroup();
        geometryGroup->setChildCount(node->mNumMeshes);

        for (unsigned int i = 0; i < node->mNumMeshes; i++)
        {
            unsigned int meshIndex = node->mMeshes[i];
            aiMesh* mesh = m_scene->mMeshes[meshIndex];
            unsigned int materialIndex = mesh->mMaterialIndex;
            Material* geometryMaterial = materials.at(materialIndex);
            optix::GeometryInstance instance = getGeometryInstance(context, geometries[meshIndex], geometryMaterial);
            geometryGroup->setChild(i, instance);

            if (dynamic_cast<DiffuseEmitter*>(geometryMaterial) != NULL)
            {
                DiffuseEmitter* emitterMaterial = (DiffuseEmitter*)(geometryMaterial);
                loadMeshLightSource(mesh, emitterMaterial);
            }
        }

        {
            optix::Acceleration acceleration = context->createAcceleration("Sbvh", "Bvh");
            acceleration->setProperty("vertex_buffer_name", "vertexBuffer");
            acceleration->setProperty("index_buffer_name", "indexBuffer");
            geometryGroup->setAcceleration(acceleration);
            acceleration->markDirty();
        }

        // Create group that contains the GeometryInstance

        optix::Group group = context->createGroup();
        group->setChildCount(1);
        group->setChild(0, geometryGroup);
        {
            optix::Acceleration acceleration = context->createAcceleration("NoAccel", "NoAccel");
            group->setAcceleration(acceleration);
        }

        return group;
    }
    else if (node->mNumChildren > 0)
    {
        QVector<optix::Group> groups;
        for (unsigned int i = 0; i < node->mNumChildren; i++)
        {
            aiNode* childNode = node->mChildren[i];
            optix::Group childGroup = getGroupFromNode(context, childNode, geometries, materials);
            if (childGroup)
            {
                groups.push_back(childGroup);
            }
        }

        if (groups.size() > 0)
        {
            optix::Group group = context->createGroup(groups.begin(), groups.end());
            optix::Acceleration acceleration = context->createAcceleration("Sbvh", "Bvh");
            group->setAcceleration(acceleration);
            return group;
        }
    }

    optix::Group emptyGroup = context->createGroup();
    optix::Acceleration acceleration = context->createAcceleration("NoAccel", "NoAccel");
    emptyGroup->setAcceleration(acceleration);
    return emptyGroup;
}

optix::GeometryInstance
Scene::getGeometryInstance(optix::Context& context, optix::Geometry& geometry, Material* material)
{
    optix::Material optix_material = material->getOptixMaterial(context);
    optix::GeometryInstance instance = context->createGeometryInstance(geometry, &optix_material, &optix_material + 1);
    material->registerGeometryInstanceValues(instance);
    return instance;
}

bool Scene::colorHasAnyComponent(const aiColor3D& color)
{
    return color.r > 0 && color.g > 0 && color.b > 0;
}

const QVector<Light>& Scene::getSceneLights() const
{
    return m_lights;
}

Camera Scene::getDefaultCamera() const
{
    return m_defaultCamera;
}

const char* Scene::getSceneName() const
{
    return m_sceneName.constData();
}

unsigned int Scene::getNumTriangles() const
{
    return m_numTriangles;
}

AAB Scene::getSceneAABB() const
{
    return m_sceneAABB;
}
