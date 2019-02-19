//
// Created by alex on 24.11.17.
//

#ifndef ORCINUS_GEOMETRYTYPES_H
#define ORCINUS_GEOMETRYTYPES_H

#include <array>
#include <vector>

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>
#include <Hors/include/HydraExport.h>

template<typename T>
inline T sqr(const T& t) {
    return t * t;
}

class ModelVertex {
    glm::vec4 Point;
    glm::vec4 Normal;
    glm::vec2 TextureCoordinates;
    uint MaterialNumber;

public:
    ModelVertex(): Point(0), Normal(0), TextureCoordinates(0), MaterialNumber(0) {}
    ModelVertex(const glm::vec4& point, const glm::vec4& normal, const glm::vec2& texCoord, const uint material):
        Point(point), Normal(normal), TextureCoordinates(texCoord), MaterialNumber(material) {}

    ModelVertex(const HydraGeomData& data, const uint i) {
        const uint index = data.getTriangleVertexIndicesArray()[i];
        Point = *reinterpret_cast<const glm::vec4*>(data.getVertexPositionsFloat4Array() + index * 4);
        Normal = glm::normalize(*reinterpret_cast<const glm::vec4*>(data.getVertexNormalsFloat4Array() + index * 4));
        TextureCoordinates = *reinterpret_cast<const glm::vec2*>(data.getVertexTexcoordFloat2Array() + index * 2);
        MaterialNumber = data.getTriangleMaterialIndicesArray()[i / 3];
    }

    glm::vec4 GetPoint() const {
        return Point;
    }

    glm::vec4 GetNormal() const {
        return Normal;
    }

    glm::vec2 GetTextureCoordinates() const {
        return TextureCoordinates;
    }

    uint GetMaterialNumber() const {
        return MaterialNumber;
    }

    const ModelVertex operator-(const ModelVertex& v) const {
        return ModelVertex(
            Point - v.Point,
            Normal - v.Normal,
            TextureCoordinates - v.TextureCoordinates,
            MaterialNumber
        );
    }

    const ModelVertex operator+(const ModelVertex& v) const {
        return ModelVertex(
            Point + v.Point,
            Normal + v.Normal,
            TextureCoordinates + v.TextureCoordinates,
            MaterialNumber
        );
    }

    const ModelVertex operator/(const float x) const {
        return ModelVertex(Point / x, Normal / x, TextureCoordinates / x, MaterialNumber);
    }

    const ModelVertex operator*(const float x) const {
        return ModelVertex(Point * x, Normal * x, TextureCoordinates * x, MaterialNumber);
    }

    bool operator==(const ModelVertex& v) const {
        static const float eps = 1e-6;
        return glm::distance(Point, v.Point) < eps
            && glm::distance(Normal, v.Normal) < eps
            && glm::distance(TextureCoordinates, v.TextureCoordinates) < eps
            && MaterialNumber == v.MaterialNumber;
    }

    void SetNormal(const glm::vec4& normal) {
        Normal = normal;
    }
};


inline float distance(const ModelVertex& a, const ModelVertex& b) {
    return glm::distance(a.GetPoint(), b.GetPoint());
}

template<typename T>
inline T sign(const T& t) {
    if (t > 0) {
        return 1;
    }
    if (t < 0) {
        return -1;
    }
    return 0;
}


class Quad {
    std::array<ModelVertex, 4> Vertices;
    glm::vec4 normal;

public:
    Quad(const ModelVertex& a, const ModelVertex& b, const ModelVertex& c, const ModelVertex& d) {
        Vertices = {a, b, c, d};
        if (Vertices[0].GetMaterialNumber() == 44) {
            normal = glm::vec4(glm::cross(glm::vec3(Vertices[3].GetPoint() - Vertices[0].GetPoint()), glm::vec3(Vertices[1].GetPoint() - Vertices[0].GetPoint())), 0);
        } else {
            normal = glm::vec4(glm::cross(glm::vec3(Vertices[1].GetPoint() - Vertices[0].GetPoint()),
                                          glm::vec3(Vertices[3].GetPoint() - Vertices[0].GetPoint())), 0);
        }
        normal = glm::normalize(normal);
    }

    std::array<ModelVertex, 4> GetVertices() const {
        return Vertices;
    }

    std::vector<Quad> Tessellate(const float minQuadSide) const {
        const auto height = distance(Vertices[0], Vertices[1]);
        const auto width = distance(Vertices[0], Vertices[3]);
        const auto heightCellsCount = static_cast<uint>(std::ceil(height / minQuadSide));
        const auto widthCellsCount = static_cast<uint>(std::ceil(width / minQuadSide));
        const auto stepHeight = (Vertices[1] - Vertices[0]) / heightCellsCount;
        const auto stepWidth = (Vertices[3] - Vertices[0]) / widthCellsCount;
        std::vector<Quad> result;
        for (uint i = 0; i < heightCellsCount; ++i) {
            for (uint j = 0; j < widthCellsCount; ++j) {
                result.emplace_back(
                    Quad(
                        Vertices[0] + stepHeight * i + stepWidth * j,
                        Vertices[0] + stepHeight * (i + 1) + stepWidth * j,
                        Vertices[0] + stepHeight * (i + 1) + stepWidth * (j + 1),
                        Vertices[0] + stepHeight * i + stepWidth * (j + 1)
                    )
                );
            }
        }
        return result;
    }

    std::vector<Quad> Tessellate(const uint inSideCount) const {
        const auto stepHeight = (Vertices[1] - Vertices[0]) / inSideCount;
        const auto stepWidth = (Vertices[3] - Vertices[0]) / inSideCount;
        std::vector<Quad> result;
        for (uint i = 0; i < inSideCount; ++i) {
            for (uint j = 0; j < inSideCount; ++j) {
                result.emplace_back(
                    Quad(
                        Vertices[0] + stepHeight * i + stepWidth * j,
                        Vertices[0] + stepHeight * (i + 1) + stepWidth * j,
                        Vertices[0] + stepHeight * (i + 1) + stepWidth * (j + 1),
                        Vertices[0] + stepHeight * i + stepWidth * (j + 1)
                    )
                );
            }
        }
        return result;
    }

    glm::vec4 GetSample(const glm::vec2& sample) const {
        return Vertices[0].GetPoint()
            + (Vertices[1] - Vertices[0]).GetPoint() * sample.x
            + (Vertices[3] - Vertices[0]).GetPoint() * sample.y;
    }

    float GetHitVectorCoef(const glm::vec4& vecBegin, const glm::vec4& vecEnd) const {
        const glm::vec4 planeNormal = Vertices[0].GetNormal();
        const glm::vec4 plane = planeNormal + glm::vec4(0, 0, 0, -glm::dot(planeNormal, Vertices[0].GetPoint()));
        const float distanceToBegin = glm::dot(plane, vecBegin);
        const float vecProjection = glm::dot(plane, vecEnd - vecBegin);
        if (std::abs(vecProjection) < 1e-6) {
            return NAN;
        }
        const float t = -distanceToBegin / vecProjection;
        return t;
    }

    bool CheckIntersectionWithVector(const glm::vec4& vecBegin, const glm::vec4& vecEnd) const {
        const float t = GetHitVectorCoef(vecBegin, vecEnd);
        if (t <= 0 || t >= 1 || std::isnan(t)) {
            return false;
        }
        const glm::vec4 pointOnPlane = vecBegin + t * (vecEnd - vecBegin);
        float square = 0;
        for (uint i = 0; i < Vertices.size(); ++i) {
            square += std::abs(glm::length(glm::cross(
                glm::vec3(Vertices[i].GetPoint() - pointOnPlane),
                glm::vec3(Vertices[(i + 1) % Vertices.size()].GetPoint() - pointOnPlane)
            )));
        }
        return std::abs(square / 2 - GetSquare()) < 1e-9;
    }

    glm::vec4 GetNormal() const {
        return normal;
        if (Vertices[0].GetMaterialNumber() == 44) {
            return glm::vec4(glm::cross(glm::vec3(Vertices[3].GetPoint() - Vertices[0].GetPoint()), glm::vec3(Vertices[1].GetPoint() - Vertices[0].GetPoint())), 0);
        }
        return glm::vec4(glm::cross(glm::vec3(Vertices[1].GetPoint() - Vertices[0].GetPoint()), glm::vec3(Vertices[3].GetPoint() - Vertices[0].GetPoint())), 0);
        return Vertices[0].GetNormal();
    }

    float GetMaxSide() const {
        return std::max(distance(Vertices[0], Vertices[1]), distance(Vertices[1], Vertices[2]));
    }

    float GetSquare() const {
        return distance(Vertices[0], Vertices[1]) * distance(Vertices[1], Vertices[2]);
    }

    void SetNormal(const glm::vec4& normal) {
        for (auto& vertex: Vertices) {
            vertex.SetNormal(normal);
        }
        this->normal = normal;
    }

    uint GetMaterialId() const {
        return Vertices[0].GetMaterialNumber();
    }

    std::pair<Quad, Quad> Split() const {
        if (distance(Vertices[0], Vertices[1]) > distance(Vertices[1], Vertices[2])) {
            return {
                {Vertices[0], (Vertices[0] + Vertices[1]) / 2.f, (Vertices[2] + Vertices[3]) / 2.f, Vertices[3]},
                {(Vertices[0] + Vertices[1]) / 2.f, Vertices[1], Vertices[2], (Vertices[2] + Vertices[3]) / 2.f}
            };
        } else {
            return {
                {Vertices[0], Vertices[1], (Vertices[1] + Vertices[2]) / 2.f, (Vertices[0] + Vertices[3]) / 2.f},
                {(Vertices[1] + Vertices[2]) / 2.f, Vertices[2], Vertices[3], (Vertices[0] + Vertices[3]) / 2.f}
            };
        }
    }
};

class QuadsContainer {
    std::vector<Quad> Quads;
    std::vector<std::pair<int, int> > Children;

public:
    void AddQuad(const Quad& quad) {
        Quads.push_back(quad);
        Children.emplace_back(-1, -1);
    }

    std::pair<int, int> SplitQuad(const int idx) {
        std::pair<int, int> result = std::make_pair<int, int>(
            static_cast<int>(Quads.size()), static_cast<int>(Quads.size()) + 1
        );
        const auto newQuads = Quads[idx].Split();
        AddQuad(newQuads.first);
        AddQuad(newQuads.second);
        Children[idx] = result;
        return result;
    }

    const Quad& GetQuad(const int idx) const {
        return Quads[idx];
    }

    int GetSize() const {
        return static_cast<int>(Quads.size());
    }

    bool HasChildren(const int idx) const {
        return Children[idx].first != -1 || Children[idx].second != -1;
    }

    bool IsFullySplittedNode(const int idx) const {
        return Children[idx].first != -1 && Children[idx].second != -1;
    }

    bool IsChildOf(const int child, const int parent) const {
        return parent != -1 && (parent == child || IsChildOf(child, Children[parent].first) || IsChildOf(child, Children[parent].second));
    }

    std::pair<int, int> GetChildren(const int idx) {
        if (Children[idx].first < 0 && Children[idx].second < 0) {
            SplitQuad(idx);
        }
        return Children[idx];
    };

    void SetChildren(const int idx, const std::pair<int, int>& newValue) {
        Children[idx] = newValue;
    }

    void RemoveQuad(const int idx) {
        Quads.erase(Quads.begin() + idx);
        if (Children[idx].first != -1 || Children[idx].second != -1) {
            exit(1);
        }
        Children.erase(Children.begin() + idx);
        for (auto& child: Children) {
            if (child.first == idx) {
                child.first = -1;
            }
            if (child.second == idx) {
                child.second = -1;
            }
        }
        for (auto& child: Children) {
            if (child.first > idx) {
                --child.first;
            }
            if (child.second > idx) {
                --child.second;
            }
        }
    }
};

std::vector<glm::vec2> GenerateRandomSamples(uint samplesNumber);

#endif //ORCINUS_GEOMETRYTYPES_H
