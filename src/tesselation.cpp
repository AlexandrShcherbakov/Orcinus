//
// Created by alex on 24.11.17.
//

#include "tesselation.h"

#include <algorithm>
#include <iostream>

std::vector<Quad> ExtractQuadsFromScene(const std::vector<HydraGeomData>& meshes) {
    std::vector<Quad> quads;
    for (auto &mesh : meshes) {
        auto meshQuads = ExtractQuadsFromScene(mesh);
        quads.insert(quads.end(), meshQuads.begin(), meshQuads.end());
    }
    return quads;
}

std::vector<Quad> ExtractQuadsFromScene(const HydraGeomData& data) {
    const uint indicesCount = data.getIndicesNumber();

    std::vector<bool> used(indicesCount / 3, false);
    std::vector<Quad> result;
    for (uint i = 0; i < indicesCount; i += 3) {
        for (uint j = i + 3; j < indicesCount; j += 3) {
            if (used[j / 3]) {
                continue;
            }
            std::vector<uint> matchedIndices;
            std::vector<uint> secondTriangleMatchedIndices;
            for (uint k = 0; k < 3; ++k) {
                for (uint h = 0; h < 3; ++h) {
                    if (data.getTriangleVertexIndicesArray()[i + k] == data.getTriangleVertexIndicesArray()[j + h]) {
                        matchedIndices.push_back(k);
                        secondTriangleMatchedIndices.push_back(h);
                    }
                }
            }
            assert(matchedIndices.size() < 3);
            if (matchedIndices.size() != 2) {
                continue;
            }
            const uint notMatchedIdx = 3 - matchedIndices[0] - matchedIndices[1];
            const ModelVertex matched1(data, i + matchedIndices[0]);
            const ModelVertex matched2(data, i + matchedIndices[1]);
            const ModelVertex unmatched(data, i + notMatchedIdx);
            if (distance(matched1, matched2) + 1e-5 < std::max(distance(matched1, unmatched), distance(matched2, unmatched))) {
                continue;
            }
            const uint forthIndex = 3 - secondTriangleMatchedIndices[0] - secondTriangleMatchedIndices[1];
            result.emplace_back(matched1, unmatched, matched2, ModelVertex(data, j + forthIndex));
            used[j / 3] = true;
        }
    }
    return result;

    std::vector<glm::vec4> planes;
    std::vector<std::vector<ModelVertex> > vertices;
    std::vector<std::vector<uint> > counters;
    for (uint i = 0; i < indicesCount; ++i) {
        const ModelVertex vertex(data, i);
        glm::vec4 plane = vertex.GetNormal() + glm::vec4(0, 0, 0, glm::dot(vertex.GetPoint(), vertex.GetNormal()));
        auto itPlane = planes.begin();
        static const float eps = 1e-6;
        for (; itPlane != planes.end() && glm::distance(plane, *itPlane) > eps; ++itPlane);
        if (itPlane == planes.end()) {
            planes.insert(itPlane, plane);
            vertices.resize(planes.size());
            counters.resize(planes.size());
            itPlane = planes.end() - 1;
        }
        const auto planeIndex = itPlane - planes.begin();
        std::vector<ModelVertex>& locVertices = vertices[planeIndex];
        std::vector<uint>& counter = counters[planeIndex];
        auto itVertex = find(locVertices.begin(), locVertices.end(), vertex);
        if (itVertex == locVertices.end()) {
            itVertex = locVertices.insert(itVertex, vertex);
            counter.push_back(0);
        }
        const auto vertexIndex = itVertex - locVertices.begin();
        counter[vertexIndex]++;
    }

    for (uint i = 0; i < planes.size(); ++i) {
        std::vector<ModelVertex>& locVertices = vertices[i];
        std::vector<uint>& counter = counters[i];
        for (int j = static_cast<int>(locVertices.size()) - 1; j >= 0; --j) {
            if (counter[j] > 2) {
                locVertices.erase(locVertices.begin() + j);
            }
        }
        assert(locVertices.size() % 4 == 0);

        float maxDist = 0;
        for (uint j = 0; j < locVertices.size(); ++j) {
            for (uint h = j + 1; h < locVertices.size(); ++h) {
                maxDist = std::max(maxDist, distance(locVertices[j], locVertices[h]));
            }
        }
        for (uint j = 0; j < locVertices.size() - 2; ++j) {
            static const float eps = 1e-6;
            if (std::abs(maxDist - distance(locVertices[j + 1], locVertices[j])) < eps) {
                std::swap(locVertices[j + 1], locVertices[j + 2]);
                break;
            }
        }
        for (uint k = 0; k < locVertices.size(); k += 4) {
            result.emplace_back(Quad(locVertices[k + 0], locVertices[k + 1], locVertices[k + 2], locVertices[k + 3]));
        }
    }

    return result;
}

std::vector<Quad> TessellateScene(const std::vector<Quad> &quads, const float MinCellWidth) {
    std::vector<Quad> result;
    for (auto& quad : quads) {
        auto subdivision = quad.Tessellate(MinCellWidth);
        result.insert(result.end(), subdivision.begin(), subdivision.end());
    }
    return result;
}

void SaveTessellation(const std::vector<Quad> &quads, const std::string &path) {
    HydraGeomData scene;
    std::vector<glm::vec4> points;
    std::vector<glm::vec4> normals;
    std::vector<glm::vec2> texCoords;
    std::vector<uint> indices;
    std::vector<uint> materials;
    for (auto& quad : quads) {
        auto vertices = quad.GetVertices();
        materials.push_back(vertices[0].GetMaterialNumber());
        materials.push_back(vertices[0].GetMaterialNumber());
        for (auto& vertex: vertices) {
            points.push_back(vertex.GetPoint());
            normals.push_back(vertex.GetNormal());
            texCoords.push_back(vertex.GetTextureCoordinates());
        }
        indices.push_back(static_cast<uint>(points.size()) - 4);
        indices.push_back(static_cast<uint>(points.size()) - 3);
        indices.push_back(static_cast<uint>(points.size()) - 2);
        indices.push_back(static_cast<uint>(points.size()) - 4);
        indices.push_back(static_cast<uint>(points.size()) - 2);
        indices.push_back(static_cast<uint>(points.size()) - 1);
    }
    scene.setData(
        static_cast<uint>(points.size()),
        reinterpret_cast<float*>(points.data()),
        reinterpret_cast<float*>(normals.data()),
        nullptr,
        reinterpret_cast<float*>(texCoords.data()),
        static_cast<uint>(indices.size()),
        indices.data(),
        materials.data()
    );
    scene.write(path);
}
