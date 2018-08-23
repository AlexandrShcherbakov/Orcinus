//
// Created by alex on 19.11.17.
//

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>

#include <Hors/include/HorsProgram.h>
#include <Hors/include/HydraExport.h>
#include <Hors/include/SceneProperties.h>
#include <Hors/include/Utils.h>

#include <GL/glut.h>

#include "formfactors.h"
#include "tesselation.h"
#include "RadiosityComputation.h"

using namespace std;

inline bool vec4Less(const glm::vec4& a, const glm::vec4& b) {
    return a.x < b.x
        || (a.x == b.x && (a.y < b.y
        || (a.y == b.y && (a.z < b.z
        || (a.z == b.z && (a.w < b.w))))));
}

class LabeledTimer {
    clock_t start;
    std::string label;
public:
    LabeledTimer(const std::string& l) {
        start = clock();
        label = l;
    }
    ~LabeledTimer() {
        auto end = static_cast<float>(clock() - start) / CLOCKS_PER_SEC * 1000;
        std::cout << label << ": " << end << " msec" << std::endl;
    }
};

class RadiosityProgram : public Hors::Program {
    std::vector<HydraGeomData> SceneMeshes;
    std::vector<Quad> quads = {};
    std::unique_ptr<Hors::SceneProperties> sceneProperties = nullptr;
    std::vector<std::map<int, float> > hierarchicalFF;
    QuadsContainer quadsHierarchy;

    Hors::GLBuffer quadPointsBuffer, quadColorsBuffer, quadNormalsBuffer;
    std::vector<Hors::GLBuffer> perMaterialIndices;
    std::vector<uint> perMaterialQuads;
    GLuint QuadRender;
    std::vector<glm::vec4> perQuadPositions, perQuadColors;
    std::vector<uint> renderedQuads;
    std::vector<glm::vec4> materialsEmission;
    std::vector<glm::vec4> materialColors;
    GLuint shadowMapTex;
    glm::vec3 lightCenter;
    float lightSide;

    void LoadFormFactorsHierarchy() {
        std::stringstream ss;
        ss << Get("DataDir") << "/" << Get("FormFactorsDir") << "/" << Get<int>("MaxHierarchyDepth") << ".bin";
        uint size;
        ifstream in(ss.str(), ios::in | ios::binary);
        if (Get<bool>("RecomputeFormFactors") || !in.good()) {
            assert(!quads.empty());
            const auto globQuads = ExtractQuadsFromScene(SceneMeshes);
            cout << "Start to compute form-factors for " << globQuads.size() << " quads" << endl;
            auto timestamp = time(nullptr);
            for (const auto& quad: globQuads) {
                quadsHierarchy.AddQuad(quad);
            }
            hierarchicalFF = FormFactorComputationEmbree(quadsHierarchy);
            std::cout << quadsHierarchy.GetSize() << ' ' << hierarchicalFF.size() << endl;
            cout << "Before compress: " << hierarchicalFF.size() << endl;
            RemoveUnnecessaryQuads(quadsHierarchy, hierarchicalFF);
            cout << "After compress: " << hierarchicalFF.size() << endl;
            std::cout << quadsHierarchy.GetSize() << ' ' << hierarchicalFF.size() << endl;
            cout << "Form-factors hierarchy computation: " << time(nullptr) - timestamp << " seconds" << endl;
            ofstream out(ss.str(), ios::out | ios::binary);
            size = static_cast<uint>(quadsHierarchy.GetSize());
            out.write(reinterpret_cast<char*>(&size), sizeof(size));
            for (int i = 0; i < quadsHierarchy.GetSize(); ++i) {
                const auto quad = quadsHierarchy.GetQuad(i);
                for (const auto& v: quad.GetVertices()) {
                    const auto point = v.GetPoint();
                    const auto normal = v.GetNormal();
                    const auto texCoord = v.GetTextureCoordinates();
                    const auto matId = v.GetMaterialNumber();
                    out.write(reinterpret_cast<const char*>(&point), sizeof(point));
                    out.write(reinterpret_cast<const char*>(&normal), sizeof(normal));
                    out.write(reinterpret_cast<const char*>(&texCoord), sizeof(texCoord));
                    out.write(reinterpret_cast<const char*>(&matId), sizeof(matId));
                }
                auto childPair = make_pair(-1, -1);
                if (quadsHierarchy.HasChildren(i)) {
                    childPair = quadsHierarchy.GetChildren(i);
                }
                out.write(reinterpret_cast<char*>(&childPair.first), sizeof(childPair.first));
                out.write(reinterpret_cast<char*>(&childPair.second), sizeof(childPair.second));
                const uint rowSize = hierarchicalFF[i].size();
                out.write(reinterpret_cast<const char*>(&rowSize), sizeof(rowSize));
                for (auto& item: hierarchicalFF[i]) {
                    out.write(reinterpret_cast<const char*>(&item.first), sizeof(item.first));
                    out.write(reinterpret_cast<char*>(&item.second), sizeof(item.second));
                }
            }
            in.close();
            out.close();
            return;
        }
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        hierarchicalFF.resize(size);
        for (uint i = 0; i < size; ++i) {
            array<ModelVertex, 4> vertices;
            for (auto &vertex : vertices) {
                glm::vec4 point;
                glm::vec4 normal;
                glm::vec2 texCoord;
                uint matId;
                in.read(reinterpret_cast<char*>(&point), sizeof(point));
                in.read(reinterpret_cast<char*>(&normal), sizeof(normal));
                in.read(reinterpret_cast<char*>(&texCoord), sizeof(texCoord));
                in.read(reinterpret_cast<char*>(&matId), sizeof(matId));
                vertex = ModelVertex(point, normal, texCoord, matId);
            }
            quadsHierarchy.AddQuad(Quad(vertices[0], vertices[1], vertices[2], vertices[3]));
            auto childPair = make_pair(-1, -1);
            in.read(reinterpret_cast<char*>(&childPair.first), sizeof(childPair.first));
            in.read(reinterpret_cast<char*>(&childPair.second), sizeof(childPair.second));
            quadsHierarchy.SetChildren(i, childPair);
            uint rowSize = 0;
            in.read(reinterpret_cast<char*>(&rowSize), sizeof(rowSize));
            for (uint j = 0; j < rowSize; ++j) {
                int key;
                float value;
                in.read(reinterpret_cast<char*>(&key), sizeof(key));
                in.read(reinterpret_cast<char*>(&value), sizeof(value));
                hierarchicalFF[i][key] = value;
            }
        }
    }

public:
    RadiosityProgram() {
        AddArgument("LoD", 1, "");
        AddArgument("MaxPatchesCount", 5000, "Max count of patches in model.");
        AddArgument("RecomputeFormFactors", false, "If true, form-factors will be recomputed force.");
        AddArgument("FormFactorsDir", "FF", "Directory with form-factors.");
        AddArgument("ScenePropertiesFile", "", "File with scene properties.");
        AddArgument("DataDir", "", "Directory with scene chunks.");
        AddArgument("MaxHierarchyDepth", 4, "");
        AddArgument("Bounces", 3, "");
    }

    void PrepareBuffers() {
        materialsEmission = sceneProperties->GetEmissionColors();
        materialColors = sceneProperties->GetDiffuseColors();
        vector<glm::vec4> firstBounce;
        {
            const auto bounces = Get<int>("Bounces");
            const auto hierarchyDepth = Get<uint>("MaxHierarchyDepth");
            float res = 0;
            const int ITERATIONS_COUNT = 1;
            for (int i = 0; i < ITERATIONS_COUNT; ++i) {
                clock_t start = clock();
                firstBounce = computeIndirectLighting(static_cast<const uint>(bounces));
                res += static_cast<float>(clock() - start) / CLOCKS_PER_SEC * 1000;
            }
            const auto result = res / ITERATIONS_COUNT;
            cout << "Compute indirect lighting 100 times: " << bounces << ' ' << hierarchyDepth << ' ' << result << endl;
        }

        std::vector<std::vector<uint> > indexBuffers;
        indexBuffers.resize(materialColors.size());
        std::vector<glm::vec4> quadsNormals;
        for (int i = 0; i < quadsHierarchy.GetSize(); ++i) {
            if (quadsHierarchy.IsFullySplittedNode(i)) {
                continue;
            }
            bool flag = false;
            for (const auto toRender : renderedQuads) {
                if (quadsHierarchy.IsChildOf(i, toRender)) {
                    flag = true;
                    break;
                }
            }
            if (flag) {
                continue;
            }
            const int matId = quadsHierarchy.GetQuad(i).GetMaterialId();
            indexBuffers[matId].push_back(static_cast<unsigned int &&>(perQuadPositions.size()));
            indexBuffers[matId].push_back(static_cast<unsigned int &&>(perQuadPositions.size() + 1));
            indexBuffers[matId].push_back(static_cast<unsigned int &&>(perQuadPositions.size() + 2));
            indexBuffers[matId].push_back(static_cast<unsigned int &&>(perQuadPositions.size()));
            indexBuffers[matId].push_back(static_cast<unsigned int &&>(perQuadPositions.size() + 2));
            indexBuffers[matId].push_back(static_cast<unsigned int &&>(perQuadPositions.size() + 3));
            glm::vec4 randVec = firstBounce[i];
            for (const auto& point: quadsHierarchy.GetQuad(i).GetVertices()) {
                perQuadPositions.push_back(point.GetPoint());
                quadsNormals.push_back(point.GetNormal());
                perQuadColors.push_back(randVec);
            }
            renderedQuads.push_back(static_cast<unsigned int &&>(i));
        }

        quadPointsBuffer = Hors::GenAndFillBuffer<GL_ARRAY_BUFFER>(perQuadPositions);
        quadColorsBuffer = Hors::GenAndFillBuffer<GL_ARRAY_BUFFER>(perQuadColors);
        quadNormalsBuffer = Hors::GenAndFillBuffer<GL_ARRAY_BUFFER>(quadsNormals);
        for (const auto &indexBuffer : indexBuffers) {
            perMaterialIndices.push_back(Hors::GenAndFillBuffer<GL_ELEMENT_ARRAY_BUFFER>(indexBuffer));
            perMaterialQuads.push_back(static_cast<unsigned int &&>(indexBuffer.size()));
        }

        QuadRender = Hors::CompileShaderProgram(
            Hors::ReadAndCompileShader("shaders/QuadRender.vert", GL_VERTEX_SHADER),
            Hors::ReadAndCompileShader("shaders/QuadRender.frag", GL_FRAGMENT_SHADER)
        );

        glUseProgram(QuadRender); CHECK_GL_ERRORS;

        GLuint QuadVAO;
        glGenVertexArrays(1, &QuadVAO); CHECK_GL_ERRORS;
        glBindVertexArray(QuadVAO); CHECK_GL_ERRORS;

        glBindBuffer(GL_ARRAY_BUFFER, *quadPointsBuffer); CHECK_GL_ERRORS;
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr); CHECK_GL_ERRORS;
        glEnableVertexAttribArray(0); CHECK_GL_ERRORS;

        glBindBuffer(GL_ARRAY_BUFFER, *quadColorsBuffer); CHECK_GL_ERRORS;
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, nullptr); CHECK_GL_ERRORS;
        glEnableVertexAttribArray(1); CHECK_GL_ERRORS;

        glBindBuffer(GL_ARRAY_BUFFER, *quadNormalsBuffer); CHECK_GL_ERRORS;
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, nullptr); CHECK_GL_ERRORS;
        glEnableVertexAttribArray(2); CHECK_GL_ERRORS;

        glEnable(GL_DEPTH_TEST); CHECK_GL_ERRORS;

        const auto cameras = sceneProperties->GetCameras(Get<Hors::WindowSize>("WindowSize").GetScreenRadio());
        assert(!cameras.empty());
        MainCamera = cameras[0];
    }

    vector<glm::vec4> computeIndirectLighting(const uint bouncesCount) {
        vector<glm::vec4> currentBounce(quadsHierarchy.GetSize(), glm::vec4(0));
        vector<glm::vec4> previousBounce(currentBounce.size(), glm::vec4(0));
        vector<glm::vec4> allBounces(currentBounce.size(), glm::vec4(0));
        for (uint i = 0; i < hierarchicalFF.size(); ++i) {
            for (const auto& f : hierarchicalFF[i]) {
                const auto materialId = quadsHierarchy.GetQuad(f.first).GetMaterialId();
                previousBounce[i] += materialsEmission[materialId] * f.second;
            }
            if (quadsHierarchy.HasChildren(i)) {
                const auto children = quadsHierarchy.GetChildren(i);
                if (children.first != -1) {
                    previousBounce[children.first] += previousBounce[i];
                }
                if (children.second != -1) {
                    previousBounce[children.second] += previousBounce[i];
                }
            }
        }
        for (uint i = 0; i < currentBounce.size(); ++i) {
            previousBounce[i] *= materialColors[quadsHierarchy.GetQuad(i).GetMaterialId()];
        }
        for (uint bounce = 0; bounce < bouncesCount; ++bounce) {
            for (uint i = 0; i < hierarchicalFF.size(); ++i) {
                for (const auto& f : hierarchicalFF[i]) {
                    currentBounce[i] += previousBounce[f.first] * f.second;
                }
                if (quadsHierarchy.HasChildren(i)) {
                    const auto children = quadsHierarchy.GetChildren(i);
                    if (children.first != -1) {
                        currentBounce[children.first] += currentBounce[i];
                    }
                    if (children.second != -1) {
                        currentBounce[children.second] += currentBounce[i];
                    }
                }
            }
            for (uint i = 0; i < currentBounce.size(); ++i) {
                currentBounce[i] *= materialColors[quadsHierarchy.GetQuad(i).GetMaterialId()];
                previousBounce[i] = currentBounce[i];
                allBounces[i] += currentBounce[i];
                currentBounce[i] = glm::vec4(0);
            }
        }
        return allBounces;
    }

    std::vector<std::map<int, float> > FormFactorComputationEmbree(QuadsContainer& quadsHierarchy) {
        std::vector<std::vector<glm::vec4> > points;
        std::vector<std::vector<uint> > indices;
        for (auto &SceneMesh : SceneMeshes) {
            std::vector<glm::vec4> meshPoints(SceneMesh.getVerticesNumber());
            for (uint k = 0; k < points.size(); ++k) {
                meshPoints[k] = glm::make_vec4(SceneMesh.getVertexPositionsFloat4Array() + k * 4);
            }
            std::vector<uint> meshIndices(
                SceneMesh.getTriangleVertexIndicesArray(),
                SceneMesh.getTriangleVertexIndicesArray() + SceneMesh.getIndicesNumber()
            );
            points.emplace_back(meshPoints);
            indices.emplace_back(meshIndices);
        }
        return ComputeFormFactorsEmbree(quadsHierarchy, points, indices, Get<uint>("MaxHierarchyDepth"));
    }

    void SetLight() {
        const auto globQuads = ExtractQuadsFromScene(SceneMeshes);
        for (const auto &globQuad : globQuads) {
            if (globQuad.GetMaterialId() == 7) { ///SCENE DEPENDENT!!!
                glm::vec4 point = globQuad.GetVertices()[0].GetPoint();
                point = glm::min(point, globQuad.GetVertices()[1].GetPoint());
                point = glm::min(point, globQuad.GetVertices()[2].GetPoint());
                point = glm::min(point, globQuad.GetVertices()[3].GetPoint());
                point.w = globQuad.GetVertices()[2].GetPoint().x - point.x;
                lightCenter = glm::vec3(point);
                lightCenter += point.w * 0.5f;
                lightSide = point.w;
                Hors::SetUniform(QuadRender, "lightPosAndSide", point);
                Hors::SetUniform(QuadRender, "lightColor", materialsEmission[globQuad.GetMaterialId()]);
                Hors::SetUniform(QuadRender, "lightNormal", globQuad.GetNormal());
            }
        }
    }

    void CreateShadowMap() {
        glGenTextures(1, &shadowMapTex); CHECK_GL_ERRORS;
        glBindTexture(GL_TEXTURE_2D, shadowMapTex); CHECK_GL_ERRORS;
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); CHECK_GL_ERRORS;
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); CHECK_GL_ERRORS;
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); CHECK_GL_ERRORS;
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); CHECK_GL_ERRORS;
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr); CHECK_GL_ERRORS;

        GLuint fb;
        glGenFramebuffers(1, &fb); CHECK_GL_ERRORS;
        glBindFramebuffer(GL_FRAMEBUFFER, fb); CHECK_GL_ERRORS;
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowMapTex, 0); CHECK_GL_ERRORS;

        GLenum status;
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER); CHECK_GL_ERRORS;
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            cerr << "Wrong framebuffer. Error: " << status << endl;
        }

        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, 1024, 1024);
        glBindFramebuffer(GL_FRAMEBUFFER, fb);

        GLuint depthRender = Hors::CompileShaderProgram(
            Hors::ReadAndCompileShader("shaders/Depth.vert", GL_VERTEX_SHADER),
            Hors::ReadAndCompileShader("shaders/Depth.frag", GL_FRAGMENT_SHADER)
        );

        const float toPlaneDist = lightSide * 0.5f;
        Hors::Camera depthCam(lightCenter + glm::vec3(0.f, toPlaneDist, 0.f) / 25.f, glm::vec3(0, -1, 0), glm::vec3(1, 0, 0),
                              std::atan2(toPlaneDist, lightSide * 0.125f / 16) * (180.0/3.141592653589793238463), 1, toPlaneDist, 100);
        Hors::SetUniform(QuadRender, "shadowMapMatrix", depthCam.GetMatrix());

        glUseProgram(depthRender); CHECK_GL_ERRORS;
        Hors::SetUniform(depthRender, "cameraMatrix", depthCam.GetMatrix());
        glClearColor(0, 0, 0, 0); CHECK_GL_ERRORS;
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); CHECK_GL_ERRORS;
        for (uint i = 0; i < perMaterialIndices.size(); ++i) {
            if (i == 7) {
                continue;
            }
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *perMaterialIndices[i]); CHECK_GL_ERRORS;
            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(perMaterialQuads[i]), GL_UNSIGNED_INT, nullptr); CHECK_GL_ERRORS;
        }
        glFinish(); CHECK_GL_ERRORS;

        glBindFramebuffer(GL_FRAMEBUFFER, 0); CHECK_GL_ERRORS;
        glDeleteFramebuffers(1, &fb); CHECK_GL_ERRORS;

        glViewport(0, 0, 1600, 900);
    }

    void Run() final {
        sceneProperties = std::make_unique<Hors::SceneProperties>(Get("ScenePropertiesFile"));
        SceneMeshes.resize(sceneProperties->GetChunksPaths().size());
        auto matrices = sceneProperties->GetMeshMatrices();
        for (uint i = 0; i < SceneMeshes.size(); ++i) {
            SceneMeshes[i].read(Get("DataDir") + "/" + sceneProperties->GetChunksPaths()[i]);
            std::vector<glm::vec4> points(SceneMeshes[i].getVerticesNumber());
            for (uint k = 0; k < points.size(); ++k) {
                points[k] = glm::make_vec4(SceneMeshes[i].getVertexPositionsFloat4Array() + k * 4);
            }
            for (auto & point: points) {
                point = point * matrices[i];
            }
            memcpy(const_cast<float*>(SceneMeshes[i].getVertexPositionsFloat4Array()), points.data(), points.size() * sizeof(points[0]));
            std::vector<glm::vec4> normals(SceneMeshes[i].getVerticesNumber());
            for (uint k = 0; k < normals.size(); ++k) {
                normals[k] = glm::make_vec4(SceneMeshes[i].getVertexNormalsFloat4Array() + k * 4);
            }
            for (auto & normal: normals) {
                normal = normal * matrices[i];
            }
            memcpy(const_cast<float*>(SceneMeshes[i].getVertexNormalsFloat4Array()), normals.data(), normals.size() * sizeof(normals[0]));
        }

        LoadFormFactorsHierarchy();
        PrepareBuffers();

        int formFactorsCount = 0;
        int maxFormFactorsCount = 0;
        for (const auto& row : hierarchicalFF) {
            formFactorsCount += row.size();
            maxFormFactorsCount = std::max(maxFormFactorsCount, static_cast<int>(row.size()));
        }
        cout << "Quads: " << hierarchicalFF.size() << endl;
        cout << "FormFactors count: " << formFactorsCount << endl;
        cout << "FormFactors per quad: " << static_cast<float>(formFactorsCount) / hierarchicalFF.size() << endl;
        cout << "Max form factors per quad: " << maxFormFactorsCount << endl;

        SetLight();
        CreateShadowMap();
    }

    void RenderFunction() final {
        glUseProgram(QuadRender); CHECK_GL_ERRORS;
        Hors::SetUniform(QuadRender, "CameraMatrix", MainCamera.GetMatrix());
        glClearColor(0, 0, 0, 0); CHECK_GL_ERRORS;
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); CHECK_GL_ERRORS;
        glBindTexture(GL_TEXTURE_2D, shadowMapTex); CHECK_GL_ERRORS;
        glActiveTexture(GL_TEXTURE0); CHECK_GL_ERRORS;
        Hors::SetUniform(QuadRender, "shadowMap", 0);
        for (uint i = 0; i < perMaterialIndices.size(); ++i) {
            Hors::SetUniform(QuadRender, "diffuseColor", materialColors[i]);
            Hors::SetUniform(QuadRender, "emissionColor", materialsEmission[i]);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *perMaterialIndices[i]); CHECK_GL_ERRORS;
            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(perMaterialQuads[i]), GL_UNSIGNED_INT, nullptr); CHECK_GL_ERRORS;
        }
        glFinish(); CHECK_GL_ERRORS;
        glutSwapBuffers();
    }
};

int main(int argc, char** argv) {
    Hors::RunProgram<RadiosityProgram>(argc, argv);
}