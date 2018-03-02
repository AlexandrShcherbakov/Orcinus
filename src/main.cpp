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
    HydraGeomData SceneData;
    std::vector<Quad> quads = {};
    std::unique_ptr<Hors::SceneProperties> sceneProperties = nullptr;
    std::vector<std::vector<glm::vec4> > initialLight;

    Hors::GLBuffer pointsBuffer, colorsBuffer, indicesBuffer;
    GLuint runSize = 0;
    std::unique_ptr<Hors::SpotLight> light;

    std::vector<std::vector<float> > LoadFormFactors() {
        std::stringstream ss;
        ss << Get("FormFactorsDir") << "/" << Get<int>("LoD") << ".bin";
        uint size;
        ifstream in(ss.str(), ios::in | ios::binary);
        if (Get<bool>("RecomputeFormFactors") || !in.good()) {
            assert(!quads.empty());
            auto ff = ComputeFormFactors(quads);
            //auto ff = ComputeAlternativeFF(quads);
            ofstream out(ss.str(), ios::out | ios::binary);
            size = static_cast<uint>(ff.size());
            out.write(reinterpret_cast<char*>(&size), sizeof(size));
            for (auto& row: ff) {
                for (auto& item: row) {
                    out.write(reinterpret_cast<char*>(&item), sizeof(item));
                }
            }
            in.close();
            out.close();
            return ff;
        }
        std::vector<std::vector<float> > ff;
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        ff.resize(size);
        for (auto& row: ff) {
            row.resize(size);
            for (auto& item: row) {
                in.read(reinterpret_cast<char*>(&item), sizeof(item));
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
    }

    void PrepareBuffers(const std::vector<std::vector<float> > & ff) {
        std::map<
            std::pair<glm::vec4, glm::vec4>,
            int,
            bool(*)(const std::pair<glm::vec4, glm::vec4>&, const std::pair<glm::vec4, glm::vec4>&)
        > pointsIndices(
            [](const std::pair<glm::vec4, glm::vec4>& a, const std::pair<glm::vec4, glm::vec4>& b) {
                return vec4Less(a.first, b.first) || (a.first == b.first && vec4Less(a.second, b.second));
            }
        );
        const auto colorsForMaterials = sceneProperties->GetDiffuseColors();
        std::vector<glm::vec4> vertices, colors;
        std::vector<uint> indices;

        for (uint quadNum = 0; quadNum < quads.size(); ++quadNum) {
            const auto quadVertices = quads[quadNum].GetVertices();
            std::array<uint, quadVertices.size()> quadIndices = {};
            //glm::vec4 color = colorsForMaterials[quadVertices[0].GetMaterialNumber()];
            //glm::vec4 color = indirectLighting[quadNum] + colorsForMaterials[quadVertices[0].GetMaterialNumber()] / 2.0f
//                + directLighting[quadNum];
            glm::vec4 color = initialLight[0][quadNum]; // / 5.0f
//                              + colorsForMaterials[quadVertices[0].GetMaterialNumber()] / 2.0f;
            for (uint i = 0; i < quadVertices.size(); ++i) {
                const auto pointWithColor = std::make_pair(quadVertices[i].GetPoint(), color);
                const auto pointIterator = pointsIndices.find(pointWithColor);
                if (pointIterator == pointsIndices.end()) {
                    pointsIndices[pointWithColor] = static_cast<uint>(vertices.size());
                    vertices.push_back(pointWithColor.first);
                    colors.push_back(pointWithColor.second);
                }
                quadIndices[i] = pointsIndices[pointWithColor];
            }
            indices.push_back(quadIndices[0]);
            indices.push_back(quadIndices[1]);
            indices.push_back(quadIndices[2]);
            indices.push_back(quadIndices[0]);
            indices.push_back(quadIndices[2]);
            indices.push_back(quadIndices[3]);
        }

        pointsBuffer = Hors::GenAndFillBuffer<GL_ARRAY_BUFFER>(vertices);
        colorsBuffer = Hors::GenAndFillBuffer<GL_ARRAY_BUFFER>(colors);
        indicesBuffer = Hors::GenAndFillBuffer<GL_ELEMENT_ARRAY_BUFFER>(indices);
        runSize = static_cast<GLuint>(indices.size());

        GLuint Program = Hors::CompileShaderProgram(
            Hors::ReadAndCompileShader("shaders/Surface.vert", GL_VERTEX_SHADER),
            Hors::ReadAndCompileShader("shaders/Surface.frag", GL_FRAGMENT_SHADER)
        );

        glUseProgram(Program); CHECK_GL_ERRORS;

        GLuint VAO;
        glGenVertexArrays(1, &VAO); CHECK_GL_ERRORS;
        glBindVertexArray(VAO); CHECK_GL_ERRORS;

        glBindBuffer(GL_ARRAY_BUFFER, *pointsBuffer); CHECK_GL_ERRORS;
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr); CHECK_GL_ERRORS;
        glEnableVertexAttribArray(0); CHECK_GL_ERRORS;

        glBindBuffer(GL_ARRAY_BUFFER, *colorsBuffer); CHECK_GL_ERRORS;
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, nullptr); CHECK_GL_ERRORS;
        glEnableVertexAttribArray(1); CHECK_GL_ERRORS;

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *indicesBuffer); CHECK_GL_ERRORS;
        glEnable(GL_DEPTH_TEST); CHECK_GL_ERRORS;

        CameraUniformLocation = glGetUniformLocation(Program, "CameraMatrix"); CHECK_GL_ERRORS;

        const auto cameras = sceneProperties->GetCameras(Get<Hors::WindowSize>("WindowSize").GetScreenRadio());
        assert(cameras.size());
        MainCamera = cameras[0];
    }

    void Run() final {
        const float MinCellWidth = 0.75f / Get<int>("LoD");

        SceneData.read(Get("InputFile"));
        quads = TesselateScene(ExtractQuadsFromScene(SceneData), MinCellWidth);
        assert(quads.size() < Get<uint>("MaxPatchesCount"));
        cout << "Start to compute form-factors for " << quads.size() << " quads" << endl;
        auto timestamp = time(nullptr);
        auto formFactors = LoadFormFactors();
        for (uint i = 0; i < formFactors.size(); ++i) {
            for (uint j = i + 1; j < formFactors[i].size(); ++j) {
                assert(formFactors[i][j] == formFactors[j][i]);
            }
        }
        cout << "Form-factors computation: " << time(nullptr) - timestamp << " seconds" << endl;

        sceneProperties = std::make_unique<Hors::SceneProperties>(Get("ScenePropertiesFile"));
        light = std::make_unique<Hors::SpotLight>(Hors::SpotLight(sceneProperties->GetLights()[0]));

        cout << "Start to compute initial light" << std::endl;
        timestamp = time(nullptr);
        initialLight = ComputeInitialLight(quads, *light);
        std::vector<glm::vec4> quadsColors(quads.size());
        auto materialColors = sceneProperties->GetSpecularColors();
        auto diffuseColors = sceneProperties->GetDiffuseColors();
        auto quadsDiffuse = std::vector<glm::vec4>(quadsColors.size());
        for (uint i = 0; i < quadsColors.size(); ++i) {
            quadsColors[i] = materialColors[quads[i].GetVertices()[0].GetMaterialNumber()];
            quadsDiffuse[i] = diffuseColors[quads[i].GetVertices()[0].GetMaterialNumber()];
        }
        initialLight = ComputeRadiosityCPU(
            VirtualBRDFTensor(quads, quadsColors, quadsDiffuse),
            formFactors,
            initialLight,
            2
        );

//        for (uint i = 1; i < initialLight.size(); ++i) {
//            for (uint j = 0; j < initialLight[i].size(); ++j) {
//                initialLight[j][0] += initialLight[j][i];
//            }
//        }
//        for (uint i = 0; i < initialLight.size(); ++i) {
//            initialLight[0][i] = initialLight[0][i] + 0.1f;// * quadsDiffuse[i];
//        }
        for (uint i = 0; i < initialLight.size(); ++i) {
            initialLight[0][i] = initialLight[i][2695] + 0.1f;
        }
        initialLight[0][2695] = glm::vec4(0, 1, 1, 0);

        for (uint i = 0; i < quads.size(); ++i) {
            if (std::abs(glm::dot(quads[i].GetNormal(), glm::vec4(1, 0, 0, 0))) == 1) {
                cout << i << endl;
            }
        }

//        for (uint i = 1; i < initialLight.size(); ++i) {
//            for (uint j = 0; j < initialLight[i].size(); ++j) {
//                initialLight[0][j] += initialLight[i][j];
////                if (initialLight[i][j] > 0 && initialLight[i][j] < 1) {
////                    initialLight[0][i] = 1;
////                    cout << i << ' ' << j << endl;
////                    break;
////                }
//            }
//        }
//        for (uint i = 0; i < initialLight.size(); ++i) {
//            initialLight[0][i] *= quadsDiffuse[i];
//        }
        cout << "Initial light computation: " << time(nullptr) - timestamp << " seconds" << endl;

        PrepareBuffers(formFactors);
    }

    void RenderFunction() final {
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUniformMatrix4fv(CameraUniformLocation, 1, GL_FALSE, glm::value_ptr(MainCamera.GetMatrix())); CHECK_GL_ERRORS;
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(runSize), GL_UNSIGNED_INT, nullptr); CHECK_GL_ERRORS;
        glFinish(); CHECK_GL_ERRORS;
        glutSwapBuffers();
    }
};

int main(int argc, char** argv) {
    Hors::RunProgram<RadiosityProgram>(argc, argv);
}