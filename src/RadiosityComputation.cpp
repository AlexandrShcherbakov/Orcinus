//
// Created by alex on 20.01.18.
//

#include <RadiosityComputation.h>

#include <iostream>


std::vector<glm::vec4> RecomputeColorsForQuadsCPU(
    const std::vector<std::vector<float> > & ff,
    const std::vector<glm::vec4> & colors,
    const std::vector<glm::vec4> & emission
) {
    const int THREADS_COUNT = 10;
    std::vector<glm::vec4> lighting(colors.size());
#pragma omp parallel for num_threads(THREADS_COUNT)
    for (uint i = 0; i < lighting.size(); ++i) {
        for (uint j = 0; j < colors.size(); ++j) {
            lighting[i] += emission[j] * ff[i][j];
        }
    }
    for (uint i = 0; i < lighting.size(); ++i) {
        lighting[i] *= colors[i];
    }
    return lighting;
}

std::vector<std::vector<float> > AdamarProduct(
    const std::vector<std::vector<float> >& m1,
    const std::vector<std::vector<float> >& m2
) {
    std::vector<std::vector<float> > result(m1);
    assert(result.size() == m1.size());
    for (uint i = 0; i < result.size(); ++i) {
        assert(result[i].size() == m2[i].size());
        for (uint j = 0; j < result[i].size(); ++j) {
            result[i][j] *= m2[i][j];
        }
    }
    return result;
}

