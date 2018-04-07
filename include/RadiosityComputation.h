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
    const std::vector<glm::vec4> & colors,
    const std::vector<glm::vec4> & emission,
    int iters = 1
);

std::vector<std::vector<float> > AdamarProduct(
    const std::vector<std::vector<float> >& m1,
    const std::vector<std::vector<float> >& m2
);


#endif //ORCINUS_RADIOSITYCOMPUTATION_H
