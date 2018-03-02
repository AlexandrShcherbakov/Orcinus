//
// Created by alex on 20.01.18.
//

#ifndef ORCINUS_RADIOSITYCOMPUTATION_H
#define ORCINUS_RADIOSITYCOMPUTATION_H

#include <GeometryTypes.h>

#include <utility>

std::vector<glm::vec4> RecomputeColorsForQuadsCPU(
    const std::vector<std::vector<float> > & ff,
    const std::vector<glm::vec4> & colors
);

class VirtualBRDFTensor {
    std::vector<Quad> Quads;
    std::vector<glm::vec4> SpecularColor;
    std::vector<glm::vec4> DiffuseColor;
    std::vector<glm::vec2> Samples;
    const uint SAMPLES_COUNT = 5;

public:
    VirtualBRDFTensor(
        std::vector<Quad> quads,
        std::vector<glm::vec4> specularColor,
        std::vector<glm::vec4> diffuseColor
    ) :
        Quads(std::move(quads)),
        SpecularColor(std::move(specularColor)),
        DiffuseColor(std::move(diffuseColor))
    {
        assert(Quads.size() == SpecularColor.size());
        assert(Quads.size() == DiffuseColor.size());
        Samples = GenerateRandomSamples(SAMPLES_COUNT);
    }

    //From i-th quad to k-th quad through j-th quad
    glm::vec3 GetValue(const int i, const int j, const int k) const {
        if (i == j || j == k) {
            return glm::vec3(0);
        }
        const auto itoj = Quads[j].GetSample(glm::vec2(0.5)) - Quads[i].GetSample(glm::vec2(0.5));
        const auto reflected = glm::normalize(glm::reflect(itoj, Quads[j].GetNormal()));
        const auto toView = glm::normalize(Quads[k].GetSample(glm::vec2(0.5)) - Quads[j].GetSample(glm::vec2(0.5)));

//        return std::max(glm::dot(reflected, toView), 0.f) * glm::vec3(SpecularColor[i]) * 100.f;
        const glm::vec3 specular = std::pow(std::max(glm::dot(reflected, toView), 0.f), SpecularColor[i].w / 100) * glm::vec3(SpecularColor[i]) * 40.f;
//        return specular;
        const glm::vec3 diffuse = glm::vec3(DiffuseColor[i]);
        const float angle = 1; glm::dot(
            glm::normalize(Quads[j].GetSample(glm::vec2(0.5)) - Quads[i].GetSample(glm::vec2(0.5))),
            glm::normalize(Quads[i].GetNormal()));
        assert(angle <= 1.f);

        return (specular + diffuse * angle) / 2.f;
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
