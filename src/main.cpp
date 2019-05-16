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
#include <chrono>

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
    decltype(std::chrono::steady_clock::now()) start;
    std::string label;
public:
    LabeledTimer(const std::string& l) {
        start = std::chrono::steady_clock::now();
        label = l;
    }
    ~LabeledTimer() {
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << label << ": " << elapsed.count() / 1000.0 << " msec" << std::endl;
    }
};

struct Label {
    std::chrono::microseconds minimum = std::chrono::microseconds::max(), maximum, sum;
    int count;
};
static map<string, Label> timers;
void printLabels() {
    for (const auto& timer : timers) {
        cout << timer.first << ": " << timer.second.sum.count() / static_cast<float>(timer.second.count) / 1000.0
             << ' ' << timer.second.minimum.count() / 1000.0 << ' ' << timer.second.maximum.count() / 1000.0 << endl;
    }
}

class LabeledTimer2 {
    decltype(std::chrono::steady_clock::now()) start;
    std::string label;
public:
    LabeledTimer2(const std::string& l) {
        start = std::chrono::steady_clock::now();
        label = l;
    }
    ~LabeledTimer2() {
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        timers[label].sum += elapsed;
        timers[label].minimum = std::min(timers[label].minimum, elapsed);
        timers[label].maximum = std::max(timers[label].maximum, elapsed);
        timers[label].count++;
    }
};

template<typename T>
void writeBin(ofstream& out, const T t) {
    out.write(reinterpret_cast<const char*>(&t), sizeof(t));
}

class RadiosityProgram : public Hors::Program {
    std::vector<HydraGeomData> SceneMeshes;
    std::vector<Quad> quads = {};
    std::unique_ptr<Hors::SceneProperties> sceneProperties = nullptr;
    std::vector<std::vector<pair<int, float>>> hierarchicalFF;
    QuadsContainer quadsHierarchy;

    Hors::GLBuffer quadPointsBuffer, quadUVBuffer, quadDirectBuffer;
    std::vector<Hors::GLBuffer> perMaterialIndices;
    std::vector<unsigned> perMaterialQuads;
    GLuint QuadRender;
    GLuint updateLightCS;
    GLuint addToMatrixCS;
    GLuint removeOldValuesCS;
    GLuint computeDoubleReflectionCS;
    GLuint computeTripleReflectionCS;
    GLuint addColumnInterReflectionCS;
    GLuint addRowInterReflectionCS;
    GLuint sumMatricesCS;
    std::vector<glm::vec4> perQuadPositions, perQuadColors;
    std::vector<unsigned> renderedQuads;
    std::vector<glm::vec4> materialsEmission;
    std::vector<glm::vec4> materialColors;
    std::vector<std::vector<glm::vec3> > multibounceMatrix;
    std::vector<glm::vec2> texCoords;
    std::vector<GLuint> textures;
    std::vector<int> textureIds;
    bool renderLines = false;
    std::vector<glm::vec3> quadCenters;
    std::vector<int> lightQuads;
    std::vector<std::vector<glm::vec3>> dynamicMatrix;
    std::vector<int> quadsInMatrix;
    std::vector<std::vector<bool>> usedQuads;
    std::vector<std::vector<std::vector<glm::vec4>>> texData;
    Hors::GLBuffer perQuadIndirect;
    Hors::GLBuffer quadsInMatrixBuffer;
    Hors::GLBuffer materialsBuffer;
    Hors::GLBuffer usedQuadsBuffer;
    Hors::GLBuffer fColumnBuffer, fRowBuffer;
    Hors::GLBuffer gColumnBuffer, gRowBuffer;
    Hors::GLBuffer localColorsBuffer;
    Hors::GLBuffer localEmissionBuffer;
    Hors::GLBuffer doubelReflection;
    std::vector<int> quadsOrder;
    std::vector<glm::vec4> directLight;
    glm::vec3 lastPos;
    GLuint localMatrixTex;

    std::vector<glm::vec4> quadsColors;

    vector<glm::vec4> lighting;

    void LoadFormFactorsHierarchy() {
        std::stringstream ss;
        ss << Get("DataDir") << "/" << Get("FormFactorsDir") << "/" << Get<int>("MaxHierarchyDepth") << ".bin";
        unsigned size;
        std::cout << ss.str() << endl;
        ifstream in(ss.str(), ios::in | ios::binary);
        if (Get<bool>("RecomputeFormFactors") || !in.good()) {
            const auto globQuads = ExtractQuadsFromScene(SceneMeshes);
            cout << "Start to compute form-factors for " << globQuads.size() << " quads" << endl;
            auto timestamp = time(nullptr);
            int i = 0;
            for (const auto& quad: globQuads) {
                quadsHierarchy.AddQuad(quad);
                ++i;
            }
            hierarchicalFF = FormFactorComputationEmbree(quadsHierarchy);
            std::cout << quadsHierarchy.GetSize() << ' ' << hierarchicalFF.size() << endl;
            cout << "Form-factors hierarchy computation: " << time(nullptr) - timestamp << " seconds" << endl;
            ofstream out(ss.str(), ios::out | ios::binary);
            size = static_cast<unsigned>(quadsHierarchy.GetSize());
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
                const unsigned rowSize = hierarchicalFF[i].size();
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
        for (unsigned i = 0; i < size; ++i) {
            array<ModelVertex, 4> vertices;
            for (auto &vertex : vertices) {
                glm::vec4 point;
                glm::vec4 normal;
                glm::vec2 texCoord;
                unsigned matId;
                in.read(reinterpret_cast<char*>(&point), sizeof(point));
                in.read(reinterpret_cast<char*>(&normal), sizeof(normal));
                in.read(reinterpret_cast<char*>(&texCoord), sizeof(texCoord));
                in.read(reinterpret_cast<char*>(&matId), sizeof(matId));
                vertex = ModelVertex(point, normal, texCoord, matId);
            }
            quadsHierarchy.AddQuad(Quad(vertices[0], vertices[1], vertices[2], vertices[3]));
            unsigned rowSize = 0;
            in.read(reinterpret_cast<char*>(&rowSize), sizeof(rowSize));
            hierarchicalFF[i].resize(rowSize);
            for (unsigned j = 0; j < rowSize; ++j) {
                int key;
                float value;
                in.read(reinterpret_cast<char*>(&key), sizeof(key));
                in.read(reinterpret_cast<char*>(&value), sizeof(value));
                hierarchicalFF[i][j] = std::make_pair(key, value);
            }
        }
    }

    void dumpFFMatrix() {
        std::stringstream ss;
        ss << Get("DataDir") << "/Dumps/FlatFF" << Get<int>("MaxHierarchyDepth") << ".bin";
        unsigned size = hierarchicalFF.size();
        ofstream out(ss.str(), ios::out | ios::binary);
        writeBin(out, size);
        for (unsigned i = 0; i < size; ++i) {
            const unsigned rowSize = hierarchicalFF[i].size();
            writeBin(out, rowSize);
            for (const auto& p : hierarchicalFF[i]) {
                writeBin(out, p.first);
                writeBin(out, p.second);
            }
        }
        out.close();
    }

    void dumpMultibounceMatrix() {
        std::stringstream ss;
        ss << Get("DataDir") << "/Dumps/MultibounceFF" << Get<int>("MaxHierarchyDepth") << ".bin";
        unsigned size = multibounceMatrix.size();
        ofstream out(ss.str(), ios::out | ios::binary);
        writeBin(out, size);
        for (unsigned i = 0; i < size; ++i) {
            for (const auto& f : multibounceMatrix[i]) {
                writeBin(out, f);
            }
        }
        out.close();
    }

    void PrepareBuffers() {
//        computeDynamicIndirectLighting();

        std::vector<std::vector<unsigned> > indexBuffers;
        indexBuffers.resize(materialColors.size());
        std::vector<glm::vec4> quadsNormals;
        for (int i = 0; i < quadsHierarchy.GetSize(); ++i) {
            const int matId = quadsHierarchy.GetQuad(i).GetMaterialId();
            indexBuffers[matId].emplace_back(perQuadPositions.size());
            indexBuffers[matId].emplace_back(static_cast<unsigned int &&>(perQuadPositions.size() + 1));
            indexBuffers[matId].emplace_back(static_cast<unsigned int &&>(perQuadPositions.size() + 2));
            indexBuffers[matId].emplace_back(static_cast<unsigned int &&>(perQuadPositions.size()));
            indexBuffers[matId].emplace_back(static_cast<unsigned int &&>(perQuadPositions.size() + 2));
            indexBuffers[matId].emplace_back(static_cast<unsigned int &&>(perQuadPositions.size() + 3));
            for (const auto& point: quadsHierarchy.GetQuad(i).GetVertices()) {
                perQuadPositions.emplace_back(point.GetPoint());
                quadsNormals.emplace_back(quadsHierarchy.GetQuad(i).GetNormal());
                perQuadColors.emplace_back(directLight[i]);
//                perQuadColors.emplace_back(point.GetNormal());
                texCoords.emplace_back(point.GetTextureCoordinates());
            }
            renderedQuads.emplace_back(static_cast<unsigned int &&>(i));
        }

//        std::vector<glm::vec4> f(directLight.size(), glm::vec4(0));
//        for (auto & i : hierarchicalFF[36641]) {
//            f[i.first] = glm::vec4(i.second);
//        }

        quadPointsBuffer = Hors::GenAndFillBuffer<GL_ARRAY_BUFFER>(perQuadPositions);
        quadUVBuffer = Hors::GenAndFillBuffer<GL_ARRAY_BUFFER>(texCoords);
        quadDirectBuffer = Hors::GenAndFillBuffer<GL_ARRAY_BUFFER>(perQuadColors);
        perQuadIndirect = Hors::GenAndFillBuffer<GL_SHADER_STORAGE_BUFFER>(lighting);
        quadsInMatrixBuffer = Hors::GenAndFillBuffer<GL_SHADER_STORAGE_BUFFER>(quadsInMatrix);
        for (const auto &indexBuffer : indexBuffers) {
            perMaterialIndices.push_back(Hors::GenAndFillBuffer<GL_ELEMENT_ARRAY_BUFFER>(indexBuffer));
            perMaterialQuads.push_back(static_cast<unsigned int &&>(indexBuffer.size()));
        }

        vector<glm::vec4> materials;
        for (int i = 0; i < quadsHierarchy.GetSize(); ++i) {
            const int matId = quadsHierarchy.GetQuad(i).GetMaterialId();
            materials.push_back(quadsColors[i]);
//            materials.push_back(materialsEmission[matId]);
            materials.push_back(directLight[i] * quadsColors[i]);
        }
        materialsBuffer = Hors::GenAndFillBuffer<GL_SHADER_STORAGE_BUFFER>(materials);

        fRowBuffer = Hors::GenAndFillBuffer<GL_SHADER_STORAGE_BUFFER>(vector<float>(Get<int>("MatrixSize")));
        fColumnBuffer = Hors::GenAndFillBuffer<GL_SHADER_STORAGE_BUFFER>(vector<float>(Get<int>("MatrixSize")));

        gRowBuffer = Hors::GenAndFillBuffer<GL_SHADER_STORAGE_BUFFER>(vector<glm::vec4>(Get<int>("MatrixSize")));
        gColumnBuffer = Hors::GenAndFillBuffer<GL_SHADER_STORAGE_BUFFER>(vector<glm::vec4>(Get<int>("MatrixSize")));

        localColorsBuffer = Hors::GenAndFillBuffer<GL_SHADER_STORAGE_BUFFER>(vector<glm::vec4>(Get<int>("MatrixSize")));

        localEmissionBuffer = Hors::GenAndFillBuffer<GL_SHADER_STORAGE_BUFFER>(vector<glm::vec4>(Get<int>("MatrixSize")));

        doubelReflection = Hors::GenAndFillBuffer<GL_SHADER_STORAGE_BUFFER>(vector<glm::vec4>(std::max(Get<int>("MatrixSize") / 1024, 1)));

        glGenTextures(1, &localMatrixTex); CHECK_GL_ERRORS;
        glBindTexture(GL_TEXTURE_2D, localMatrixTex); CHECK_GL_ERRORS;
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, Get<int>("MatrixSize"), Get<int>("MatrixSize")); CHECK_GL_ERRORS;

        QuadRender = Hors::CompileShaderProgram(
            Hors::ReadAndCompileShader("shaders/QuadRender.vert", GL_VERTEX_SHADER),
            Hors::ReadAndCompileShader("shaders/QuadRender.frag", GL_FRAGMENT_SHADER)
        );

        glUseProgram(QuadRender); CHECK_GL_ERRORS;

        std::stringstream ss;
        ss << Get<int>("MatrixSize");
        std::map<string, string> replacement;
        replacement["MATRIX_SIZE_VALUE"] = ss.str();
        cout << replacement["MATRIX_SIZE_VALUE"] << ' ' << Get<int>("MatrixSize") << endl;

        GLuint QuadVAO;
        glGenVertexArrays(1, &QuadVAO); CHECK_GL_ERRORS;
        glBindVertexArray(QuadVAO); CHECK_GL_ERRORS;

        glBindBuffer(GL_ARRAY_BUFFER, *quadPointsBuffer); CHECK_GL_ERRORS;
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr); CHECK_GL_ERRORS;
        glEnableVertexAttribArray(0); CHECK_GL_ERRORS;

        glBindBuffer(GL_ARRAY_BUFFER, *quadDirectBuffer); CHECK_GL_ERRORS;
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, nullptr); CHECK_GL_ERRORS;
        glEnableVertexAttribArray(1); CHECK_GL_ERRORS;

        glBindBuffer(GL_ARRAY_BUFFER, *quadUVBuffer); CHECK_GL_ERRORS;
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, nullptr); CHECK_GL_ERRORS;
        glEnableVertexAttribArray(2); CHECK_GL_ERRORS;

        glEnable(GL_DEPTH_TEST); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *perQuadIndirect); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, *perQuadIndirect); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;


        updateLightCS = Hors::CompileComputeShaderProgram(
                Hors::ReadAndCompileShader("shaders/UpdateLighting.comp", GL_COMPUTE_SHADER, replacement)
        );
        glUseProgram(updateLightCS); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *perQuadIndirect); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, *perQuadIndirect); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *quadsInMatrixBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, *quadsInMatrixBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *materialsBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, *materialsBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *localEmissionBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, *localEmissionBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindImageTexture(0, localMatrixTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA16F);CHECK_GL_ERRORS;

        usedQuadsBuffer = Hors::GenAndFillBuffer<GL_SHADER_STORAGE_BUFFER>(std::vector<int>(Get<int>("MatrixSize")));

        removeOldValuesCS = Hors::CompileComputeShaderProgram(
                Hors::ReadAndCompileShader("shaders/RemoveOldValues.comp", GL_COMPUTE_SHADER)
        );
        glUseProgram(removeOldValuesCS); CHECK_GL_ERRORS;

        glBindImageTexture(0, localMatrixTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA16F); CHECK_GL_ERRORS;

        computeDoubleReflectionCS = Hors::CompileComputeShaderProgram(
                Hors::ReadAndCompileShader("shaders/ComputeDoubleReflection.comp", GL_COMPUTE_SHADER, replacement)
        );
        glUseProgram(computeDoubleReflectionCS); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *quadsInMatrixBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, *quadsInMatrixBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *materialsBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, *materialsBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *fRowBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, *fRowBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *fColumnBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, *fColumnBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *doubelReflection); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, *doubelReflection); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *localColorsBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, *localColorsBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *localEmissionBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, *localEmissionBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        addColumnInterReflectionCS = Hors::CompileComputeShaderProgram(
                Hors::ReadAndCompileShader("shaders/AddColumnInterReflection.comp", GL_COMPUTE_SHADER, replacement)
        );
        glUseProgram(addColumnInterReflectionCS); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *fColumnBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, *fColumnBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *gColumnBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, *gColumnBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *localColorsBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, *localColorsBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindImageTexture(0, localMatrixTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA16F); CHECK_GL_ERRORS;

        addRowInterReflectionCS = Hors::CompileComputeShaderProgram(
                Hors::ReadAndCompileShader("shaders/AddRowInterReflection.comp", GL_COMPUTE_SHADER, replacement)
        );
        glUseProgram(addRowInterReflectionCS); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *fRowBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, *fRowBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *gRowBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, *gRowBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *localColorsBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, *localColorsBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindImageTexture(0, localMatrixTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA16F); CHECK_GL_ERRORS;

        sumMatricesCS = Hors::CompileComputeShaderProgram(
                Hors::ReadAndCompileShader("shaders/SumMatrices.comp", GL_COMPUTE_SHADER, replacement)
        );
        glUseProgram(sumMatricesCS); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *usedQuadsBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, *usedQuadsBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *gRowBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, *gRowBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *gColumnBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, *gColumnBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *localColorsBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, *localColorsBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindImageTexture(0, localMatrixTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA16F); CHECK_GL_ERRORS;

        addToMatrixCS = Hors::CompileComputeShaderProgram(
                Hors::ReadAndCompileShader("shaders/AddToMatrix.comp", GL_COMPUTE_SHADER, replacement)
        );
        glUseProgram(addToMatrixCS); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *usedQuadsBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, *usedQuadsBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *fRowBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, *fRowBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *fColumnBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, *fColumnBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *doubelReflection); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, *doubelReflection); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *gRowBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, *gRowBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *gColumnBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, *gColumnBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *localColorsBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, *localColorsBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindImageTexture(0, localMatrixTex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA16F); CHECK_GL_ERRORS;

        computeTripleReflectionCS = Hors::CompileComputeShaderProgram(
                Hors::ReadAndCompileShader("shaders/ComputeTripleReflection.comp", GL_COMPUTE_SHADER, replacement)
        );
        glUseProgram(computeTripleReflectionCS); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *fRowBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, *fRowBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *fColumnBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, *fColumnBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *doubelReflection); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, *doubelReflection); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *gRowBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, *gRowBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *gColumnBuffer); CHECK_GL_ERRORS;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, *gColumnBuffer); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;
    }

    void computeIndirectLighting(const unsigned bouncesCount) {
        LabeledTimer timer("Full radiosity");
        vector<glm::vec4> prevBounce(quadsHierarchy.GetSize(), glm::vec4(0));
        vector<glm::vec4> bounce(quadsHierarchy.GetSize(), glm::vec4(0));
        directLight.resize(quadsHierarchy.GetSize());
        for (int i = 0; i < quadsHierarchy.GetSize(); ++i) {
            lighting[i] = materialsEmission[quadsHierarchy.GetQuad(i).GetMaterialId()];
            prevBounce[i] = lighting[i];
        }
        for (unsigned k = 0; k < bouncesCount; ++k) {
            for (unsigned i = 0; i < hierarchicalFF.size(); ++i) {
                for (const auto& it : hierarchicalFF[i]) {
                    bounce[i] += prevBounce[it.first] * it.second;
                }
            }
            for (unsigned i = 0; i < hierarchicalFF.size(); ++i) {
                if (k != 0) {
                    lighting[i] += bounce[i];
                } else {
//                    lighting[i] += bounce[i];
                    directLight[i] = bounce[i];
                }
                prevBounce[i] = bounce[i] * quadsColors[i];
                bounce[i] = glm::vec4(0);
            }
        }
    }

    void computeDynamicIndirectLighting() {
        for (unsigned i = 0; i < dynamicMatrix.size(); ++i) {
            lighting[quadsInMatrix[i]] = glm::vec4(0);
            const auto targetId = quadsHierarchy.GetQuad(quadsInMatrix[i]).GetMaterialId();
            for (unsigned j = 0; j < dynamicMatrix[i].size(); ++j) {
                const auto materialId = quadsHierarchy.GetQuad(quadsInMatrix[j]).GetMaterialId();
                lighting[quadsInMatrix[i]] += materialsEmission[materialId] * glm::make_vec4(dynamicMatrix[i][j]);
            }
            lighting[quadsInMatrix[i]] *= materialColors[targetId];
            lighting[quadsInMatrix[i]] += materialsEmission[targetId];
        }
    }

    std::vector<std::vector<pair<int, float>>> FormFactorComputationEmbree(QuadsContainer& quadsHierarchy) {
        std::vector<std::vector<glm::vec4> > points;
        std::vector<std::vector<unsigned> > indices;
        for (auto &SceneMesh : SceneMeshes) {
            std::vector<glm::vec4> meshPoints(SceneMesh.getVerticesNumber());
            for (unsigned k = 0; k < meshPoints.size(); ++k) {
                meshPoints[k] = glm::make_vec4(SceneMesh.getVertexPositionsFloat4Array() + k * 4);
            }
            std::vector<unsigned> meshIndices(
                SceneMesh.getTriangleVertexIndicesArray(),
                SceneMesh.getTriangleVertexIndicesArray() + SceneMesh.getIndicesNumber()
            );
            points.emplace_back(meshPoints);
            indices.emplace_back(meshIndices);
        }
        return ComputeFormFactorsEmbree(quadsHierarchy, points, indices, Get<unsigned>("MaxHierarchyDepth"));
    }

    void MultiplyMatrices(std::vector<std::vector<glm::vec3> >& a, std::vector<std::vector<glm::vec3> > bTransposed) const {
        const unsigned size = a.size();
        const auto originCopy = a;
        for (unsigned i = 0; i < size; ++i) {
            for (unsigned j = 0; j < size; ++j) {
                a[i][j] = glm::vec3(0);
                for (unsigned k = 0; k < size; ++k) {
                    a[i][j] += originCopy[i][k] * bTransposed[j][k];
                }
            }
        }
    }

    void ComputeMultibounceMatrix() {
        std::vector<std::vector<glm::vec3> > coloredFF(hierarchicalFF.size());
        for (unsigned i = 0; i < coloredFF.size(); ++i) {
            coloredFF[i].assign(hierarchicalFF.size(), glm::vec3(0));
            for (unsigned j = 0; j < coloredFF[i].size(); ++j) {
                //FF type was changed
//                if (hierarchicalFF[i].count(j)) {
//                    coloredFF[i][j] = hierarchicalFF[j][i] * glm::vec3(materialColors[quadsHierarchy.GetQuad(j).GetMaterialId()]);
//                }
            }
        }
        multibounceMatrix.assign(hierarchicalFF.size(), std::vector<glm::vec3>(hierarchicalFF.size(), glm::vec3(0)));
        for (int bounce = 0; bounce < Get<int>("Bounces"); ++bounce) {
            if (bounce) {
                MultiplyMatrices(multibounceMatrix, coloredFF);
            }
            for (unsigned i = 0; i < multibounceMatrix.size(); ++i) {
                multibounceMatrix[i][i] += 1;
            }
        }
    }

    void ComputeMultibounceMatrix_v2() {
        multibounceMatrix.assign(1, std::vector<glm::vec3>(1, glm::vec3(1)));
        std::vector<glm::vec3> colors(hierarchicalFF.size());
        for (unsigned i = 0; i < colors.size(); ++i) {
            colors[i] = glm::vec3(materialColors[quadsHierarchy.GetQuad(i).GetMaterialId()]);
        }
        for (unsigned i = 1; i < hierarchicalFF.size(); ++i) {
            std::vector<glm::vec3> fColumn(multibounceMatrix.size());
            std::vector<glm::vec3> fRow(multibounceMatrix.size());

            //hierarchicalFF type was changed.
//            for (unsigned j = 0; j < fColumn.size(); ++j) {
//                fColumn[j] = glm::vec3(hierarchicalFF[j][i]);
//                fRow[j] = glm::vec3(hierarchicalFF[i][j]);
//            }
            std::vector<glm::vec3> gColumn(fColumn), gRow(fRow);
            glm::vec3 doubleReflection(0);
            for (unsigned j = 0; j < fColumn.size(); ++j) {
                doubleReflection += fRow[j] * fColumn[j] * colors[j];
            }
            for (unsigned j = 0; j < fColumn.size(); ++j) {
                gColumn[j] += fColumn[j] * colors[i] * doubleReflection;
                gRow[j] += fRow[j] * colors[i] * doubleReflection;
            }
            std::vector<glm::vec3> interReflection(fColumn.size(), glm::vec3(0));
            for (unsigned j = 0; j < fColumn.size(); ++j) {
                for (unsigned k = 0; k < fColumn.size(); ++k) {
                    interReflection[j] += multibounceMatrix[j][k] * fColumn[k] * colors[k];
                }
                gColumn[j] += interReflection[j];
            }
            interReflection.assign(fColumn.size(), glm::vec3(0));
            for (unsigned j = 0; j < fColumn.size(); ++j) {
                for (unsigned k = 0; k < fColumn.size(); ++k) {
                    interReflection[j] += multibounceMatrix[k][j] * fRow[k] * colors[k];
                }
                gRow[j] += interReflection[j];
            }
            for (unsigned j = 0; j < fColumn.size(); ++j) {
                for (unsigned k = 0; k < fRow.size(); ++k) {
                    multibounceMatrix[j][k] += gColumn[j] * gRow[k] * colors[i];
                }
            }
            for (unsigned j = 0; j < fColumn.size(); ++j) {
                multibounceMatrix[j].push_back(gColumn[j]);
            }
            multibounceMatrix.push_back(gRow);
            multibounceMatrix.back().push_back(glm::vec3(1));
            for (unsigned j = 0; j < gColumn.size(); ++j) {
                for (unsigned k = 0; k < gColumn.size(); ++k) {
                    multibounceMatrix.back().back() += gColumn[j] * gRow[k] * colors[k];
                }
            }
        }
    }

    GLuint CreateTex(const Hors::SceneProperties::TextureRecordInfo& record) {
        ifstream in(Get("DataDir") + "/" + record.location, ios::in | ios::binary);
        assert(in.is_open());
        in.seekg(record.offset);
        texData.resize(texData.size() + 1);
        texData.back().resize(record.height);
        std::vector<unsigned char> data(record.bytesize);
        for (unsigned i = 0; i < record.height; ++i) {
            texData.back()[i].resize(record.width);
            for (unsigned j = 0; j < record.width; ++j) {
                for (unsigned k = 0; k < 4; ++k) {
                    in.read(reinterpret_cast<char *>(&data[(i * record.width + j) * 4 + k]), sizeof(char));
                    texData.back()[i][j][k] = data[(i * record.width + j) * 4 + k] / 255.f;
                }
            }
        }

        GLuint texId;
        glGenTextures(1, &texId); CHECK_GL_ERRORS;
        glActiveTexture(GL_TEXTURE0 + record.id); CHECK_GL_ERRORS;
        glBindTexture(GL_TEXTURE_2D, texId); CHECK_GL_ERRORS;
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, record.width, record.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data()); CHECK_GL_ERRORS;
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); CHECK_GL_ERRORS;
        glGenerateMipmap(GL_TEXTURE_2D); CHECK_GL_ERRORS;
        return texId;
    }

    void CreateTextures() {
        const auto records = sceneProperties->GetTextures();
        for (const auto& rec: records) {
            textures.push_back(CreateTex(rec));
        }
    }

    void InitDynamicMatrix() {
        dynamicMatrix.assign(1, std::vector<glm::vec3>(1, glm::vec3(1)));
        for (int i = 1; i < Get<int>("MatrixSize"); ++i) {
            std::vector<glm::vec3> fColumn(dynamicMatrix.size());
            std::vector<glm::vec3> fRow(dynamicMatrix.size());

            for (unsigned j = 0; j < fColumn.size(); ++j) {
//                fColumn[j] = glm::vec3(hierarchicalFF[quadsInMatrix[j]][quadsInMatrix[i]]);
//                fRow[j] = glm::vec3(hierarchicalFF[quadsInMatrix[i]][quadsInMatrix[j]]);
            }
            std::vector<glm::vec3> gColumn(fColumn), gRow(fRow);
            glm::vec3 doubleReflection(0);
            for (unsigned j = 0; j < fColumn.size(); ++j) {
                doubleReflection += fRow[j] * fColumn[j] * glm::vec3(quadsColors[quadsInMatrix[j]]);
            }
            for (unsigned j = 0; j < fColumn.size(); ++j) {
                gColumn[j] += fColumn[j] * glm::vec3(quadsColors[quadsInMatrix[i]]) * doubleReflection;
                gRow[j] += fRow[j] * glm::vec3(quadsColors[quadsInMatrix[i]]) * doubleReflection;
            }
            for (unsigned j = 0; j < fColumn.size(); ++j) {
                for (unsigned k = 0; k < fColumn.size(); ++k) {
                    gColumn[j] += dynamicMatrix[j][k] * fColumn[k] * glm::vec3(quadsColors[quadsInMatrix[k]]);
                }
            }
            for (unsigned j = 0; j < fColumn.size(); ++j) {
                for (unsigned k = 0; k < fColumn.size(); ++k) {
                    gRow[j] += dynamicMatrix[k][j] * fRow[k] * glm::vec3(quadsColors[quadsInMatrix[k]]);
                }
            }
            for (unsigned j = 0; j < fColumn.size(); ++j) {
                for (unsigned k = 0; k < fRow.size(); ++k) {
                    dynamicMatrix[j][k] += gColumn[j] * gRow[k] * glm::vec3(quadsColors[quadsInMatrix[i]]);
                }
            }
            for (unsigned j = 0; j < fColumn.size(); ++j) {
                dynamicMatrix[j].push_back(gColumn[j]);
            }
            dynamicMatrix.push_back(gRow);
            dynamicMatrix.back().push_back(glm::vec3(1));
            for (unsigned j = 0; j < gColumn.size(); ++j) {
                for (unsigned k = 0; k < gColumn.size(); ++k) {
                    dynamicMatrix.back().back() += gColumn[j] * gRow[k] * glm::vec3(quadsColors[quadsInMatrix[k]]);
                }
            }
        }
    }

    std::pair<int, float> FarthestIncludedDistance() {
        LabeledTimer2 timer("FindToExclude");
        float dist = 0;
        int idx = 0;
        for (unsigned i = 0; i < quadsInMatrix.size(); ++i) {
            if (find(lightQuads.begin(), lightQuads.end(), quadsInMatrix[i]) != lightQuads.end()) {
                continue;
            }
            const float newDist = glm::distance(quadCenters[quadsInMatrix[i]], MainCamera.GetPosition());
            if (newDist > dist) {
                dist = newDist;
                idx = i;
            }
        }
        return make_pair(idx, dist);
    }

    pair<int, float> NearestExcludedDistance() {
        LabeledTimer2 timer("FindToInclude");
        int ret = quadsOrder[0];
        quadsOrder.erase(quadsOrder.begin());
        return make_pair(ret, glm::distance(quadCenters[ret], MainCamera.GetPosition()));
        float dist = 1e6;
        int idx = 0;
        for (unsigned i = 0; i < quadCenters.size(); ++i) {
            if (std::find(quadsInMatrix.begin(), quadsInMatrix.end(), i) != quadsInMatrix.end()) {
                continue;
            }
            const float newDist = glm::distance(quadCenters[i], MainCamera.GetPosition());
            if (newDist < dist) {
                dist = newDist;
                idx = i;
            }
        }
        return make_pair(idx, dist);
    }

    void AddToMatrix(const int idx, const unsigned place) {
        LabeledTimer2 timer("AddToMatrix");

        std::vector<float> fRowToBuffer;
        std::vector<float> fColumnToBuffer;
        fRowToBuffer.resize(Get<int>("MatrixSize"));
        fColumnToBuffer.resize(Get<int>("MatrixSize"));

        for (int i = 0; i < hierarchicalFF[idx].size(); ++i) {
            auto it = std::find(quadsInMatrix.begin(), quadsInMatrix.end(), hierarchicalFF[idx][i].first);
            if (it == quadsInMatrix.end()) {
                continue;
            }
            fColumnToBuffer[it - quadsInMatrix.begin()] = hierarchicalFF[idx][i].second;
            fRowToBuffer[it - quadsInMatrix.begin()] = hierarchicalFF[idx][i].second;
        }
//        for (int j = 0; j < Get<int>("MatrixSize"); ++j) {
//            fColumnToBuffer.push_back(hierarchicalFF[quadsInMatrix[j]][idx]);
//            fRowToBuffer.push_back(hierarchicalFF[idx][quadsInMatrix[j]]);
//        }

        std::vector<int> usedToBuffer;
        usedToBuffer.reserve(Get<int>("MatrixSize"));
        for (int j = 0; j < Get<int>("MatrixSize"); ++j) {
            usedToBuffer.push_back(usedQuads[quadsInMatrix[j]][idx] ? 1 : 0);
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *quadsInMatrixBuffer); CHECK_GL_ERRORS;
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, place * sizeof(idx), sizeof(idx), &idx); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *fRowBuffer); CHECK_GL_ERRORS;
        glBufferData(GL_SHADER_STORAGE_BUFFER, fRowToBuffer.size() * sizeof(fRowToBuffer[0]), fRowToBuffer.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *fColumnBuffer); CHECK_GL_ERRORS;
        glBufferData(GL_SHADER_STORAGE_BUFFER, fColumnToBuffer.size() * sizeof(fColumnToBuffer[0]), fColumnToBuffer.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, *usedQuadsBuffer); CHECK_GL_ERRORS;
        glBufferData(GL_SHADER_STORAGE_BUFFER, usedToBuffer.size() * sizeof(usedToBuffer[0]), usedToBuffer.data(), GL_STATIC_DRAW); CHECK_GL_ERRORS;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;

        glFlush(); CHECK_GL_ERRORS;

        {
            LabeledTimer2 timer("RemoveOldValues");
            Hors::SetUniform(removeOldValuesCS, "place", place);

            glUseProgram(removeOldValuesCS); CHECK_GL_ERRORS;

            glMemoryBarrier(GL_ALL_BARRIER_BITS); CHECK_GL_ERRORS;
            glDispatchCompute(std::max(Get<int>("MatrixSize") / 1024, 1), 1, 1); CHECK_GL_ERRORS;
            glMemoryBarrier(GL_ALL_BARRIER_BITS); CHECK_GL_ERRORS;
        }

        {
            LabeledTimer2 timer("DoubleReflection");
            Hors::SetUniform(computeDoubleReflectionCS, "place", place);

            glUseProgram(computeDoubleReflectionCS); CHECK_GL_ERRORS;

            glMemoryBarrier(GL_ALL_BARRIER_BITS); CHECK_GL_ERRORS;
            glDispatchCompute(std::max(Get<int>("MatrixSize") / 1024, 1), 1, 1); CHECK_GL_ERRORS;
            glMemoryBarrier(GL_ALL_BARRIER_BITS); CHECK_GL_ERRORS;
        }

        {
            LabeledTimer2 timer("TripleReflection");

            glUseProgram(computeTripleReflectionCS); CHECK_GL_ERRORS;

            glMemoryBarrier(GL_ALL_BARRIER_BITS); CHECK_GL_ERRORS;
            glDispatchCompute(std::max(Get<int>("MatrixSize") / 1024, 1), 1, 1); CHECK_GL_ERRORS;
            glMemoryBarrier(GL_ALL_BARRIER_BITS); CHECK_GL_ERRORS;
        }

        {
            LabeledTimer2 timer("AddColumnInterReflection");

            glUseProgram(addColumnInterReflectionCS); CHECK_GL_ERRORS;

            glMemoryBarrier(GL_ALL_BARRIER_BITS); CHECK_GL_ERRORS;
            glDispatchCompute(Get<int>("MatrixSize"), 1, 1); CHECK_GL_ERRORS;
            glMemoryBarrier(GL_ALL_BARRIER_BITS); CHECK_GL_ERRORS;
        }

        {
            LabeledTimer2 timer("AddRowInterReflection");

            glUseProgram(addRowInterReflectionCS); CHECK_GL_ERRORS;

            glMemoryBarrier(GL_ALL_BARRIER_BITS); CHECK_GL_ERRORS;
            glDispatchCompute(Get<int>("MatrixSize"), 1, 1); CHECK_GL_ERRORS;
            glMemoryBarrier(GL_ALL_BARRIER_BITS); CHECK_GL_ERRORS;
        }

        {
            LabeledTimer2 timer("sumMatrices");
            Hors::SetUniform(sumMatricesCS, "place", place);

            glUseProgram(sumMatricesCS); CHECK_GL_ERRORS;

            glMemoryBarrier(GL_ALL_BARRIER_BITS); CHECK_GL_ERRORS;
            if (Get<int>("MatrixSize") >= 4096) {
                glDispatchCompute(Get<int>("MatrixSize") / 1024, 1, 1); CHECK_GL_ERRORS;
            } else {
                glDispatchCompute(Get<int>("MatrixSize") / 32, Get<int>("MatrixSize") / 32, 1); CHECK_GL_ERRORS;
            }
            glMemoryBarrier(GL_ALL_BARRIER_BITS); CHECK_GL_ERRORS;
        }

        {
            LabeledTimer2 timer("AddToMatrixCS");
            Hors::SetUniform(addToMatrixCS, "place", place);

            glUseProgram(addToMatrixCS); CHECK_GL_ERRORS;

            glMemoryBarrier(GL_ALL_BARRIER_BITS); CHECK_GL_ERRORS;
            glDispatchCompute(1, 1, 1); CHECK_GL_ERRORS;
            glMemoryBarrier(GL_ALL_BARRIER_BITS); CHECK_GL_ERRORS;
        }
//        glFinish(); CHECK_GL_ERRORS;
    }

    void UpdateLightBuffer() {
        LabeledTimer2 timer("UpdateLightBuffer");
        glUseProgram(updateLightCS); CHECK_GL_ERRORS;
        glMemoryBarrier(GL_ALL_BARRIER_BITS); CHECK_GL_ERRORS;
        if (Get<int>("MatrixSize") >= 4096) {
            glDispatchCompute(Get<int>("MatrixSize") / 1024, 1, 1); CHECK_GL_ERRORS;
        } else {
            glDispatchCompute(Get<int>("MatrixSize"), 1, 1); CHECK_GL_ERRORS;
        }
        glMemoryBarrier(GL_ALL_BARRIER_BITS); CHECK_GL_ERRORS;
    }

    void UpdateMatrixInfo(const int idx, const unsigned place) {
        LabeledTimer2 timer("UpdateMatrixInfo");
        usedQuads[quadsInMatrix[place]].assign(usedQuads[quadsInMatrix[place]].size(), false);
        for (unsigned i = 0; i < quadsInMatrix.size(); ++i) {
            if (i == place) {
                continue;
            }
            usedQuads[quadsInMatrix[i]][idx] = true;
            usedQuads[idx][quadsInMatrix[i]] = true;
        }
        quadsInMatrix[place] = idx;
    }

    void RecomputeLighting() {
        if (glm::length(lastPos - MainCamera.GetPosition()) > 1e-5) {
            updateQuadsOrder();
        }
        const auto includedDist = FarthestIncludedDistance();
        const auto excludedDist = NearestExcludedDistance();
        lastPos = MainCamera.GetPosition();
        static int z = 0;
        if (excludedDist.second >= includedDist.second) {
            if (z) {
                printLabels();
                z = 0;
            }
            return;
        }
        LabeledTimer2 timer("RecomputeLighting");
        AddToMatrix(excludedDist.first, includedDist.first);
        UpdateMatrixInfo(excludedDist.first, includedDist.first);
        UpdateLightBuffer();
        ++z;
    }

    void ComputeQuadColors() {
        for (int i = 0; i < quadsHierarchy.GetSize(); ++i) {
            const int materialId = quadsHierarchy.GetQuad(i).GetMaterialId();
            quadsColors.emplace_back(materialColors[materialId]);
            int textureId = textureIds[materialId];
            if (textureId == -1) {
                continue;
            }
            float topUV = 1e6f;
            float bottomUV = -1e6f;
            float leftUV = 1e6f;
            float rightUV = -1e6f;
            for (const auto& vert : quadsHierarchy.GetQuad(i).GetVertices()) {
                topUV = std::min(topUV, vert.GetTextureCoordinates().y);
                bottomUV = std::max(bottomUV, vert.GetTextureCoordinates().y);
                leftUV = std::min(leftUV, vert.GetTextureCoordinates().x);
                rightUV = std::max(rightUV, vert.GetTextureCoordinates().x);
            }
            const int top = static_cast<int>(topUV * texData[textureId].size()) % (texData[textureId].size() + 1);
            const int bottom = static_cast<int>(bottomUV * texData[textureId].size()) % (texData[textureId].size() + 1);
            const int left = static_cast<int>(leftUV * texData[textureId][0].size()) % (texData[textureId][0].size() + 1);
            const int right = static_cast<int>(rightUV * texData[textureId][0].size()) % (texData[textureId][0].size() + 1);
            assert(top < bottom);
            assert(left < right);
            glm::vec4 texColor(0);
            for (int j = top; j < bottom; ++j) {
                for (int k = left; k < right; ++k) {
                    texColor += glm::pow(texData[textureId][j][k], glm::vec4(2.2f));
                }
            }
            texColor /= (bottom - top) * (right - left);
            quadsColors.back() *= texColor;
        }
    }

    void updateQuadsOrder() {
        quadsOrder.resize(hierarchicalFF.size());
        for (unsigned int i = 0; i < quadsOrder.size(); ++i) {
            quadsOrder[i] = i;
        }

        std::sort(quadsOrder.begin(), quadsOrder.end(), [this](int i, int j){
            float inMatrixCorrectionI = 0;
            float inMatrixCorrectionJ = 0;
            if (quadCenters[i].y < 0.3  || std::find(quadsInMatrix.begin(), quadsInMatrix.end(), i) != quadsInMatrix.end()) {
                inMatrixCorrectionI = 1e10;
            }
            if (quadCenters[j].y < 0.3 || std::find(quadsInMatrix.begin(), quadsInMatrix.end(), j) != quadsInMatrix.end()) {
                inMatrixCorrectionJ = 1e10;
            }
            return glm::distance(quadCenters[i], MainCamera.GetPosition()) + inMatrixCorrectionI < glm::distance(quadCenters[j], MainCamera.GetPosition()) + inMatrixCorrectionJ;
        });
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
        AddArgument("MatrixSize", 512, "");
    }

    void Run() final {
        sceneProperties = std::make_unique<Hors::SceneProperties>(Get("ScenePropertiesFile"));
        CreateTextures();
        textureIds = sceneProperties->GetDiffuseTextures();
        auto matrices = sceneProperties->GetMeshMatrices();
        SceneMeshes.resize(matrices.size());
        for (unsigned i = 0; i < SceneMeshes.size(); ++i) {
            SceneMeshes[i].read(Get("DataDir") + "/" + sceneProperties->GetChunksPaths()[matrices[i].second]);
            std::vector<glm::vec4> points(SceneMeshes[i].getVerticesNumber());
            for (unsigned k = 0; k < points.size(); ++k) {
                points[k] = glm::make_vec4(SceneMeshes[i].getVertexPositionsFloat4Array() + k * 4);
            }
            for (auto & point: points) {
                point = point * matrices[i].first;
            }
            memcpy(const_cast<float*>(SceneMeshes[i].getVertexPositionsFloat4Array()), points.data(), points.size() * sizeof(points[0]));
            std::vector<glm::vec4> normals(SceneMeshes[i].getVerticesNumber());
            for (unsigned k = 0; k < normals.size(); ++k) {
                normals[k] = glm::make_vec4(SceneMeshes[i].getVertexNormalsFloat4Array() + k * 4);
            }
            for (auto & normal: normals) {
                normal = normal * matrices[i].first;
            }
            memcpy(const_cast<float*>(SceneMeshes[i].getVertexNormalsFloat4Array()), normals.data(), normals.size() * sizeof(normals[0]));
        }

        LoadFormFactorsHierarchy();

        materialsEmission = sceneProperties->GetEmissionColors();
//        for (auto &v: materialsEmission) {
//            v *= 10;
//        }
        materialColors = sceneProperties->GetDiffuseColors();
        const auto cameras = sceneProperties->GetCameras(Get<Hors::WindowSize>("WindowSize").GetScreenRadio());
        assert(!cameras.empty());
        MainCamera = cameras[0];

        for (int i = 0; i < quadsHierarchy.GetSize(); ++i) {
            quadCenters.emplace_back(glm::vec3(quadsHierarchy.GetQuad(i).GetSample(glm::vec2(0.5, 0.5))));
            if (glm::length(glm::vec3(materialsEmission[quadsHierarchy.GetQuad(i).GetMaterialId()])) > 1e-5f) {
                lightQuads.emplace_back(i);
            }
        }

        ComputeQuadColors();

        lighting.assign(static_cast<unsigned long>(quadsHierarchy.GetSize()), glm::vec4(0));
        computeIndirectLighting(3);

        quadsInMatrix.resize(static_cast<unsigned long>(quadsHierarchy.GetSize()));
        for (unsigned i = 0; i < quadsInMatrix.size(); ++i) {
            quadsInMatrix[i] = i;
        }
        std::sort(quadsInMatrix.begin(), quadsInMatrix.end(), [this](int i, int j) {
//            return glm::distance(quadCenters[i], MainCamera.GetPosition()) < glm::distance(quadCenters[j], MainCamera.GetPosition());
            const bool light1 = std::find(lightQuads.begin(), lightQuads.end(), i) != lightQuads.end();
            const bool light2 = std::find(lightQuads.begin(), lightQuads.end(), j) != lightQuads.end();

            return (light1 && !light2)
                   || ((light1 == light2) && i < j);// && glm::distance(quadCenters[i], MainCamera.GetPosition()) < glm::distance(quadCenters[j], MainCamera.GetPosition()));
        });
        quadsInMatrix.resize(Get<int>("MatrixSize"));

        PrepareBuffers();
        quadsOrder.resize(hierarchicalFF.size());
        updateQuadsOrder();

//        InitDynamicMatrix();
        usedQuads.resize(static_cast<unsigned long>(quadsHierarchy.GetSize()));
        for (auto &usedQuad : usedQuads) {
            usedQuad.assign(static_cast<unsigned long>(quadsHierarchy.GetSize()), false);
        }
        for (int i : quadsInMatrix) {
            for (int j : quadsInMatrix) {
                usedQuads[i][j] = true;
            }
        }

        cout << "Dynamic matrix prepared" << endl;

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

        Hors::SetUniform(updateLightCS, "dynamicMatrixSize", Get<int>("MatrixSize"));

        AddKeyboardEvent('z', [this](){ renderLines = !renderLines; });
        AddKeyboardEvent('p', [this](){ std::cout << MainCamera.GetPosition().x << ' ' << MainCamera.GetPosition().y << ' ' << MainCamera.GetPosition().z << std::endl; });
        AddKeyboardEvent('m', [this](){
            MainCamera.SetPosition(glm::vec3(-0.207217, 0.356178, -0.0529972));
//            MainCamera.SetPosition(glm::vec3(7.77363, 6.95324, -24.4086));
//            MainCamera = cameras[0];
        });
        AddKeyboardEvent('l', [this](){
            int z = 0;
            for (unsigned int i = 0; i < hierarchicalFF.size(); ++i) {
                const int matId = quadsHierarchy.GetQuad(i).GetMaterialId();
                if (glm::distance(quadCenters[i], MainCamera.GetPosition()) < glm::distance(quadCenters[z], MainCamera.GetPosition()) && materialsEmission[matId].x > 1e-5) {
                    z = i;
                }
            }
            cout << z << endl;
            glm::vec4 randColor((float)rand() / (1 << 16), (float)rand() / (1 << 16), (float)rand() / (1 << 16), 1);
            randColor *= 10;
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, *materialsBuffer); CHECK_GL_ERRORS;
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, (2 * z + 1) * sizeof(randColor), sizeof(randColor), &randColor); CHECK_GL_ERRORS;
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); CHECK_GL_ERRORS;
            UpdateLightBuffer();
        });
    }

    void RenderFunction() final {
        RecomputeLighting();
        glUseProgram(QuadRender); CHECK_GL_ERRORS;
        Hors::SetUniform(QuadRender, "CameraMatrix", MainCamera.GetMatrix());
        glClearColor(0, 0, 0, 0); CHECK_GL_ERRORS;
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); CHECK_GL_ERRORS;
        glPolygonMode(GL_FRONT_AND_BACK, renderLines ? GL_LINE : GL_FILL);
        for (unsigned i = 0; i < perMaterialIndices.size(); ++i) {
            Hors::SetUniform(QuadRender, "Tex", textureIds[i] != -1 ? textureIds[i] : 0);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *perMaterialIndices[i]); CHECK_GL_ERRORS;
            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(perMaterialQuads[i]), GL_UNSIGNED_INT, nullptr); CHECK_GL_ERRORS;
        }
        glFinish(); CHECK_GL_ERRORS;
        glutSwapBuffers();
    }
};

int main(int argc, char** argv) {
    Hors::RunProgram<RadiosityProgram>(argc, argv);
    printLabels();
}