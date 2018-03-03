//
// Created by alex on 20.01.18.
//

#ifndef ORCINUS_RADIOSITYCOMPUTATION_H
#define ORCINUS_RADIOSITYCOMPUTATION_H

#include <GeometryTypes.h>

#include <cmath>
#include <utility>

std::vector<glm::vec4> RecomputeColorsForQuadsCPU(
    const std::vector<std::vector<float> > & ff,
    const std::vector<glm::vec4> & colors
);

inline float random(const float from, const float to) {
    return static_cast<float>(rand()) / RAND_MAX * (to - from) + from;
}

class VirtualBRDFTensor {
    std::vector<glm::vec4> SpecularColor;
    std::vector<glm::vec4> DiffuseColor;
    std::vector<glm::vec3> Normals;
    std::vector<glm::vec3> Centers;

public:
    VirtualBRDFTensor(
        std::vector<Quad> quads,
        std::vector<glm::vec4> specularColor,
        std::vector<glm::vec4> diffuseColor
    ) :
        SpecularColor(std::move(specularColor)),
        DiffuseColor(std::move(diffuseColor))
    {
        assert(Quads.size() == SpecularColor.size());
        assert(Quads.size() == DiffuseColor.size());
        Normals.resize(quads.size());
        Centers.resize(quads.size());
        for (uint i = 0; i < quads.size(); ++i) {
            Normals[i] = glm::vec3(quads[i].GetNormal());
            Centers[i] = glm::vec3(quads[i].GetSample(glm::vec2(0.5)));
        }
    }


    //From i-th quad to k-th quad through j-th quad
    glm::vec3 GetValue(const int i, const int j, const int k) const {
        if (i == j || j == k) {
            return glm::vec3(0);
        }
        const auto itoj = Centers[j] - Centers[i];
        const auto reflected = glm::normalize(glm::reflect(itoj, Normals[j]));
        const auto toView = glm::normalize(Centers[k] - Centers[j]);

        const glm::vec3 specular = std::pow(std::max(glm::dot(reflected, toView), 0.f), SpecularColor[j].w)
            * glm::vec3(SpecularColor[j]) * SpecularColor[j].w;
        const glm::vec3 diffuse = glm::vec3(DiffuseColor[j]) * 3.14f;

        return (specular + diffuse) / 2.f;
    }

};

std::vector<std::vector<float> > AdamarProduct(
    const std::vector<std::vector<float> >& m1,
    const std::vector<std::vector<float> >& m2
);

std::vector<std::vector<glm::vec4> > ComputeRadiosityCPU(
    const VirtualBRDFTensor& rho,
    const std::vector<std::vector<float> >& ff,
    const std::vector<std::vector<glm::vec4> >& initLight,
    int iter=3
);

#endif //ORCINUS_RADIOSITYCOMPUTATION_H
