//
// Created by alex on 20.01.18.
//

#ifndef ORCINUS_RADIOSITYCOMPUTATION_H
#define ORCINUS_RADIOSITYCOMPUTATION_H

#include <GeometryTypes.h>

#include <cmath>
#include <fstream>
#include <utility>

std::vector<glm::vec4> RecomputeColorsForQuadsCPU(
    const std::vector<std::vector<float> > & ff,
    const std::vector<glm::vec4> & colors
);

class VirtualBRDFTensor {
    std::vector<glm::vec4> DiffuseColor;
    std::map<std::array<short, 3>, glm::vec3> specularSparseTensor;

public:
    VirtualBRDFTensor(
        const std::string& filename,
        const std::vector<Quad> &quads,
        const std::vector<glm::vec4> &specularColor,
        std::vector<glm::vec4> diffuseColor,
        const std::vector<std::vector<float> > &ff
    ) :
        DiffuseColor(std::move(diffuseColor))
    {
        uint size;
        std::ifstream in(filename, std::ios::in | std::ios::binary);
        if (!in.good()) {
            assert(!quads.empty());
            ComputeTensor(quads, specularColor, ff);
            std::ofstream out(filename, std::ios::out | std::ios::binary);
            size = static_cast<uint>(specularSparseTensor.size());
            out.write(reinterpret_cast<char*>(&size), sizeof(size));
            for (const auto& value: specularSparseTensor) {
                for (const auto idx: value.first) {
                    out.write(reinterpret_cast<const char*>(&idx), sizeof(idx));
                }
                out.write(reinterpret_cast<const char*>(&(value.second)), sizeof(value.second));
            }
            in.close();
            out.close();
            return;
        }
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        for (uint i = 0; i < size; ++i) {
            std::array<short, 3> key = {};
            glm::vec3 value;
            for (auto & idx: key) {
                in.read(reinterpret_cast<char*>(&idx), sizeof(idx));
            }
            in.read(reinterpret_cast<char*>(&(value)), sizeof(value));
            specularSparseTensor[key] = value;
        }
        in.close();
    }

    void ComputeTensor(
        const std::vector<Quad> &quads,
        const std::vector<glm::vec4> &specularColor,
        const std::vector<std::vector<float>> &ff
    ) {
        std::vector<glm::vec3> Normals;
        std::vector<glm::vec3> Centers;
        assert(quads.size() == specularColor.size());
        assert(quads.size() == DiffuseColor.size());
        const auto SIZE = static_cast<const int>(quads.size());

        Normals.resize(static_cast<unsigned long>(SIZE));
        Centers.resize(static_cast<unsigned long>(SIZE));
        for (int i = 0; i < SIZE; ++i) {
            Normals[i] = glm::vec3(quads[i].GetNormal());
            Centers[i] = glm::vec3(quads[i].GetSample(glm::vec2(0.5)));
        }

        const int THREADS_COUNT = 10;
#pragma omp parallel for num_threads(THREADS_COUNT)
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                if (!ff[i][j]) {
                    continue;
                }
                for (int k = 0; k < SIZE; ++k) {
                    if (!ff[j][k]) {
                        continue;
                    }
                    const auto itoj = Centers[j] - Centers[i];
                    const auto reflected = normalize(reflect(itoj, Normals[j]));
                    const auto toView = normalize(Centers[k] - Centers[j]);

                    const glm::vec3 specular = std::pow(std::max(dot(reflected, toView), 0.f), specularColor[j].w)
                        * glm::vec3(specularColor[j]) * specularColor[j].w;
                    if (length(specular) < 1e-7f) {
                        continue;
                    }
                    std::array<short, 3> key = {static_cast<short>(i), static_cast<short>(j), static_cast<short>(k)};
                    specularSparseTensor[key] = specular;
                }
            }
        }
    }

    glm::vec4 GetDiffuse(const int quadIndex) const {
        return DiffuseColor[quadIndex];
    }

    std::map<std::array<short, 3>, glm::vec3> GetSpecularSparseTensor() const {
        return specularSparseTensor;
    };
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
