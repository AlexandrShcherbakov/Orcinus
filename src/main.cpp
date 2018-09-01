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

    Hors::GLBuffer quadPointsBuffer, quadColorsBuffer;
    std::vector<Hors::GLBuffer> perMaterialIndices;
    std::vector<uint> perMaterialQuads;
    GLuint QuadRender;
    std::vector<glm::vec4> perQuadPositions, perQuadColors;
    std::vector<uint> renderedQuads;
    std::vector<glm::vec4> materialsEmission;
    std::vector<glm::vec4> materialColors;
    std::vector<std::vector<glm::vec4> > multibounceMatrix;

    void LoadFormFactorsHierarchy() {
        std::stringstream ss;
        ss << Get("DataDir") << "/" << Get("FormFactorsDir") << "/" << Get<int>("MaxHierarchyDepth") << ".bin";
        uint size;
        ifstream in(ss.str(), ios::in | ios::binary);
        if (Get<bool>("RecomputeFormFactors") || !in.good()) {
            const auto globQuads = ExtractQuadsFromScene(SceneMeshes);
            cout << "Start to compute form-factors for " << globQuads.size() << " quads" << endl;
            auto timestamp = time(nullptr);
            for (const auto& quad: globQuads) {
                quadsHierarchy.AddQuad(quad);
            }
            hierarchicalFF = FormFactorComputationEmbree(quadsHierarchy);
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

    void PrepareBuffers() {
        materialsEmission = sceneProperties->GetEmissionColors();
        materialColors = sceneProperties->GetDiffuseColors();

        {
            LabeledTimer timer("Multibounce matrix computation");
            ComputeMultibounceMatrix();
        }

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

        glEnable(GL_DEPTH_TEST); CHECK_GL_ERRORS;

        const auto cameras = sceneProperties->GetCameras(Get<Hors::WindowSize>("WindowSize").GetScreenRadio());
        assert(!cameras.empty());
        MainCamera = cameras[0];
    }

    vector<glm::vec4> computeIndirectLighting(const uint bouncesCount) {
        vector<glm::vec4> allBounces(multibounceMatrix.size(), glm::vec4(0));
        for (uint i = 0; i < multibounceMatrix.size(); ++i) {
            for (uint j = 0; j < multibounceMatrix[i].size(); ++j) {
                const auto materialId = quadsHierarchy.GetQuad(j).GetMaterialId();
                allBounces[i] += materialsEmission[materialId] * multibounceMatrix[i][j];
            }
        }
        return allBounces;
    }

    std::vector<std::map<int, float> > FormFactorComputationEmbree(QuadsContainer& quadsHierarchy) {
        std::vector<std::vector<glm::vec4> > points;
        std::vector<std::vector<uint> > indices;
        for (auto &SceneMesh : SceneMeshes) {
            std::vector<glm::vec4> meshPoints(SceneMesh.getVerticesNumber());
            for (uint k = 0; k < meshPoints.size(); ++k) {
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

    void MultiplyMatrices(std::vector<std::vector<glm::vec4> >& a, std::vector<std::vector<glm::vec4> > bTransposed) const {
        const uint size = a.size();
        const auto originCopy = a;
        for (uint i = 0; i < size; ++i) {
            for (uint j = 0; j < size; ++j) {
                a[i][j] = glm::vec4(0);
                for (uint k = 0; k < size; ++k) {
                    a[i][j] += originCopy[i][k] * bTransposed[j][k];
                }
            }
        }
    }

    void ComputeMultibounceMatrix() {
        std::vector<std::vector<glm::vec4> > coloredFF(hierarchicalFF.size());
        for (uint i = 0; i < coloredFF.size(); ++i) {
            coloredFF[i].assign(hierarchicalFF.size(), glm::vec4(0));
            for (uint j = 0; j < coloredFF[i].size(); ++j) {
                if (hierarchicalFF[i].count(j)) {
                    coloredFF[i][j] = hierarchicalFF[j][i] * materialColors[quadsHierarchy.GetQuad(j).GetMaterialId()];
                }
            }
        }
        multibounceMatrix.assign(hierarchicalFF.size(), std::vector<glm::vec4>(hierarchicalFF.size(), glm::vec4(0)));
        for (int bounce = 0; bounce < Get<int>("Bounces"); ++bounce) {
            if (bounce) {
                MultiplyMatrices(multibounceMatrix, coloredFF);
            }
            for (uint i = 0; i < multibounceMatrix.size(); ++i) {
                multibounceMatrix[i][i] += 1;
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
    }

    void RenderFunction() final {
        glUseProgram(QuadRender); CHECK_GL_ERRORS;
        Hors::SetUniform(QuadRender, "CameraMatrix", MainCamera.GetMatrix());
        glClearColor(0, 0, 0, 0); CHECK_GL_ERRORS;
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); CHECK_GL_ERRORS;
        for (uint i = 0; i < perMaterialIndices.size(); ++i) {
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