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

class RadiosityProgram : public Hors::Program {
    std::vector<HydraGeomData> SceneMeshes;
    std::vector<Quad> quads = {};
    std::unique_ptr<Hors::SceneProperties> sceneProperties = nullptr;
    std::vector<glm::vec4> indirectLight;
    GLuint Program;

    Hors::GLBuffer pointsBuffer, indirectLightBuffer, diffuseColorBuffer, indicesBuffer;
    Hors::GLBuffer normalsBuffer;
    GLuint runSize = 0;
    Hors::GLBuffer quadPointsBuffer, quadColorsBuffer, quadIndicesBuffer;
    GLuint QuadRender;

    std::vector<std::map<uint, float> > LoadFormFactors() {
        std::stringstream ss;
        ss << Get("DataDir") << "/" << Get("FormFactorsDir") << "/" << Get<int>("LoD") << ".bin";
        uint size;
        ifstream in(ss.str(), ios::in | ios::binary);
        if (Get<bool>("RecomputeFormFactors") || !in.good()) {
            assert(!quads.empty());
            auto ff = FormFactorComputationEmbree();
            ofstream out(ss.str(), ios::out | ios::binary);
            size = static_cast<uint>(ff.size());
            out.write(reinterpret_cast<char*>(&size), sizeof(size));
            for (auto& row: ff) {
                const uint rowSize = row.size();
                out.write(reinterpret_cast<const char*>(&rowSize), sizeof(rowSize));
                for (auto& item: row) {
                    out.write(reinterpret_cast<char*>(&item), sizeof(item));
                }
            }
            in.close();
            out.close();
            return ff;
        }
        std::vector<std::map<uint, float> > ff;
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        ff.resize(size);
        for (auto& row: ff) {
            uint rowSize;
            in.read(reinterpret_cast<char*>(&rowSize), sizeof(rowSize));
            std::pair<uint, float> item;
            for (uint i = 0; i < rowSize; ++i) {
                in.read(reinterpret_cast<char*>(&item), sizeof(item));
                row[item.first] = item.second;
            }
        }
        in.close();
        return ff;
    }

public:
    RadiosityProgram() {
        AddArgument("LoD", 1, "");
        AddArgument("MaxPatchesCount", 5000, "Max count of patches in model.");
        AddArgument("RecomputeFormFactors", false, "If true, form-factors will be recomputed force.");
        AddArgument("FormFactorsDir", "FF", "Directory with form-factors.");
        AddArgument("ScenePropertiesFile", "", "File with scene properties.");
        AddArgument("DataDir", "", "Directory with scene chunks.");
    }

    void PrepareBuffers(const std::vector<glm::vec4>& colorsPerQuad) {
        std::map<
            std::pair<glm::vec4, glm::vec4>,
            int,
            bool(*)(const std::pair<glm::vec4, glm::vec4>&, const std::pair<glm::vec4, glm::vec4>&)
        > pointsIndices(
            [](const std::pair<glm::vec4, glm::vec4>& a, const std::pair<glm::vec4, glm::vec4>& b) {
                return vec4Less(a.first, b.first) || (a.first == b.first && vec4Less(a.second, b.second));
            }
        );
        std::vector<glm::vec4> vertices, indirectLightDeviceBuffer;
        std::vector<glm::vec4> diffusePerVertex;
        std::vector<uint> indices, interpolationValuesCount;
        std::vector<glm::vec4> normalsPerVertex;

        std::vector<glm::vec4> perQuadPositions, perQuadColors;
        std::vector<uint> perQuadIndices;

        for (uint quadNum = 0; quadNum < quads.size(); ++quadNum) {
            const auto quadVertices = quads[quadNum].GetVertices();
            std::array<uint, quadVertices.size()> quadIndices = {};
            glm::vec4 indirectQuadLight = indirectLight[quadNum];
            for (uint i = 0; i < quadVertices.size(); ++i) {
                const auto pointWithNormal = std::make_pair(quadVertices[i].GetPoint(), quadVertices[i].GetNormal());
                const auto pointIterator = pointsIndices.find(pointWithNormal);
                if (pointIterator == pointsIndices.end()) {
                    pointsIndices[pointWithNormal] = static_cast<uint>(vertices.size());
                    vertices.push_back(pointWithNormal.first);
                    indirectLightDeviceBuffer.push_back(indirectQuadLight);
                    interpolationValuesCount.push_back(1);
                    diffusePerVertex.push_back(colorsPerQuad[quadNum]);
                    normalsPerVertex.push_back(quads[quadNum].GetNormal());
                } else {
                    indirectLightDeviceBuffer[pointsIndices[pointWithNormal]] += indirectQuadLight;
                    interpolationValuesCount[pointsIndices[pointWithNormal]]++;
                }
                quadIndices[i] = pointsIndices[pointWithNormal];
            }
            indices.push_back(quadIndices[0]);
            indices.push_back(quadIndices[1]);
            indices.push_back(quadIndices[2]);
            indices.push_back(quadIndices[0]);
            indices.push_back(quadIndices[2]);
            indices.push_back(quadIndices[3]);

            perQuadIndices.push_back(perQuadPositions.size());
            perQuadIndices.push_back(perQuadPositions.size() + 1);
            perQuadIndices.push_back(perQuadPositions.size() + 2);
            perQuadIndices.push_back(perQuadPositions.size());
            perQuadIndices.push_back(perQuadPositions.size() + 2);
            perQuadIndices.push_back(perQuadPositions.size() + 3);
            for (const auto& point: quads[quadNum].GetVertices()) {
                perQuadPositions.push_back(point.GetPoint());
                perQuadColors.push_back(colorsPerQuad[quadNum]);
            }
        }
        for (uint i = 0; i < indirectLightDeviceBuffer.size(); ++i) {
            indirectLightDeviceBuffer[i] /= interpolationValuesCount[i];
        }

        pointsBuffer = Hors::GenAndFillBuffer<GL_ARRAY_BUFFER>(vertices);
        indirectLightBuffer = Hors::GenAndFillBuffer<GL_ARRAY_BUFFER>(indirectLightDeviceBuffer);
        diffuseColorBuffer = Hors::GenAndFillBuffer<GL_ARRAY_BUFFER>(diffusePerVertex);
        normalsBuffer = Hors::GenAndFillBuffer<GL_ARRAY_BUFFER>(normalsPerVertex);

        quadPointsBuffer = Hors::GenAndFillBuffer<GL_ARRAY_BUFFER>(perQuadPositions);
        quadColorsBuffer = Hors::GenAndFillBuffer<GL_ARRAY_BUFFER>(perQuadColors);
        quadIndicesBuffer = Hors::GenAndFillBuffer<GL_ELEMENT_ARRAY_BUFFER>(perQuadIndices);

        indicesBuffer = Hors::GenAndFillBuffer<GL_ELEMENT_ARRAY_BUFFER>(indices);
        runSize = static_cast<GLuint>(indices.size());

        Program = Hors::CompileShaderProgram(
            Hors::ReadAndCompileShader("shaders/Radiosity.vert", GL_VERTEX_SHADER),
            Hors::ReadAndCompileShader("shaders/Radiosity.frag", GL_FRAGMENT_SHADER)
        );

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

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *quadIndicesBuffer); CHECK_GL_ERRORS;
        glEnable(GL_DEPTH_TEST); CHECK_GL_ERRORS;

//        glUseProgram(Program); CHECK_GL_ERRORS;
//
//        GLuint VAO;
//        glGenVertexArrays(1, &VAO); CHECK_GL_ERRORS;
//        glBindVertexArray(VAO); CHECK_GL_ERRORS;
//
//        glBindBuffer(GL_ARRAY_BUFFER, *pointsBuffer); CHECK_GL_ERRORS;
//        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr); CHECK_GL_ERRORS;
//        glEnableVertexAttribArray(0); CHECK_GL_ERRORS;
//
//        glBindBuffer(GL_ARRAY_BUFFER, *indirectLightBuffer); CHECK_GL_ERRORS;
//        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, nullptr); CHECK_GL_ERRORS;
//        glEnableVertexAttribArray(1); CHECK_GL_ERRORS;
//
//        glBindBuffer(GL_ARRAY_BUFFER, *diffuseColorBuffer); CHECK_GL_ERRORS;
//        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, nullptr); CHECK_GL_ERRORS;
//        glEnableVertexAttribArray(2); CHECK_GL_ERRORS;
//
//        glBindBuffer(GL_ARRAY_BUFFER, *normalsBuffer); CHECK_GL_ERRORS;
//        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, nullptr); CHECK_GL_ERRORS;
//        glEnableVertexAttribArray(3); CHECK_GL_ERRORS;
//
//        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *indicesBuffer); CHECK_GL_ERRORS;
//        glEnable(GL_DEPTH_TEST); CHECK_GL_ERRORS;
//
//        CameraUniformLocation = glGetUniformLocation(Program, "CameraMatrix"); CHECK_GL_ERRORS;

        const auto cameras = sceneProperties->GetCameras(Get<Hors::WindowSize>("WindowSize").GetScreenRadio());
        assert(!cameras.empty());
        MainCamera = cameras[0];
    }

    std::vector<std::map<uint, float> > FormFactorComputationEmbree() {
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
        return ComputeFormFactorsEmbree(quads, points, indices);
    }

    void Run() final {
        const float MinCellWidth = 0.75f / Get<int>("LoD");

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
        quads = TessellateScene(ExtractQuadsFromScene(SceneMeshes), MinCellWidth);

        assert(quads.size() < Get<uint>("MaxPatchesCount"));

        cout << "Start to compute form-factors for " << quads.size() << " quads" << endl;
        auto timestamp = time(nullptr);
        auto formFactors = LoadFormFactors();
        cout << "Form-factors computation: " << time(nullptr) - timestamp << " seconds" << endl;

        auto materialsEmission = sceneProperties->GetEmissionColors();
        std::vector<glm::vec4> emissionPerQuad(quads.size());
        std::transform(
            quads.begin(),
            quads.end(),
            emissionPerQuad.begin(),
            [&materialsEmission](const Quad &q) { return materialsEmission[q.GetMaterialId()]; });

        auto materialColors = sceneProperties->GetDiffuseColors();
        std::vector<glm::vec4> colorsPerQuad(quads.size());
        std::transform(
            quads.begin(),
            quads.end(),
            colorsPerQuad.begin(),
            [&materialColors](const Quad &q) { return materialColors[q.GetMaterialId()]; });

        const auto clusters = HierarchicalClusterization(formFactors);
        unsigned long maxCluster = 0, minCluster = formFactors.size();
        for (const auto& cluster: clusters) {
            maxCluster = std::max(maxCluster, cluster.size());
            minCluster = std::min(minCluster, cluster.size());
        }
        std::cout << "Max cluster size: " << maxCluster << std::endl;
        std::cout << "Min cluster size: " << minCluster << std::endl;
        std::cout << "Clusters count: " << clusters.size() << std::endl;
//
//        auto modifiedMatrix = SimplifyMatrixUsingClasters(formFactors, clusters);
//        float mse = 0;
//        float maxAbsoluteError = 0;
//        for (uint i = 0; i < formFactors.size(); ++i) {
//            for (uint j = 0; j < formFactors[i].size(); ++j) {
//                mse += sqr(formFactors[i][j] - modifiedMatrix[i][j]);
//                maxAbsoluteError = std::max(maxAbsoluteError, std::abs(formFactors[i][j] - modifiedMatrix[i][j]));
//            }
//        }
//        std::cout << "MSE: " << mse / sqr(formFactors.size()) << std::endl;
//        std::cout << "Max absolute error: " << maxAbsoluteError << std::endl;

        indirectLight = RecomputeColorsForQuadsCPU(formFactors, colorsPerQuad, emissionPerQuad, 4);

        std::vector<glm::vec4> quadClusterColors(quads.size());
        for (const auto& cluster : clusters) {
            const glm::vec4 clusterColor = Hors::GenRandomColor();
            for (const auto quadId: cluster) {
                quadClusterColors[quadId] = clusterColor;
            }
        }
        PrepareBuffers(indirectLight);
    }

    void RenderFunction() final {
        glUseProgram(QuadRender); CHECK_GL_ERRORS;
        Hors::SetUniform(QuadRender, "CameraMatrix", MainCamera.GetMatrix());
        glClearColor(0, 0, 0, 0); CHECK_GL_ERRORS;
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); CHECK_GL_ERRORS;
//        glUniformMatrix4fv(CameraUniformLocation, 1, GL_FALSE, glm::value_ptr(MainCamera.GetMatrix())); CHECK_GL_ERRORS;
//        Hors::SetUniform(Program, "cameraPosition", glm::vec4(MainCamera.GetPosition(), 1)); CHECK_GL_ERRORS;
//        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(runSize), GL_UNSIGNED_INT, nullptr); CHECK_GL_ERRORS;
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(quads.size() * 6), GL_UNSIGNED_INT, nullptr); CHECK_GL_ERRORS;
        glFinish(); CHECK_GL_ERRORS;
        glutSwapBuffers();
    }
};

int main(int argc, char** argv) {
    Hors::RunProgram<RadiosityProgram>(argc, argv);
}