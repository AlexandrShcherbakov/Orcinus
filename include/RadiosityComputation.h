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
    const std::vector<std::map<uint, float> > & ff,
    const std::vector<glm::vec4> & colors,
    const std::vector<glm::vec4> & emission,
    int iters = 1
);

void RemoveUnnecessaryQuads(
    QuadsContainer& quads,
    std::vector<std::map<int, float> > & ff
);

std::vector<glm::vec4> RecomputeColorsForQuadsCPU(
    const QuadsContainer& quads,
    const std::vector<std::map<uint, float> > & ff,
    const std::vector<glm::vec4> & colors,
    const std::vector<glm::vec4> & emission,
    int iters = 1
);

#endif //ORCINUS_RADIOSITYCOMPUTATION_H
