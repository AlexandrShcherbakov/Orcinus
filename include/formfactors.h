//
// Created by alex on 25.11.17.
//

#ifndef ORCINUS_FORMFACTORS_H
#define ORCINUS_FORMFACTORS_H

#include <vector>
#include <Hors/include/SceneProperties.h>

#include "GeometryTypes.h"


std::vector<std::vector<float> > ComputeFormFactors(
    const std::vector<Quad>& quads,
    bool verbose=true,
    bool checkIntersections=true,
    bool antiradiance=false
);

std::vector<std::vector<float> > ComputeFormFactorsEmbree(
    const std::vector<Quad>& quads,
    const std::vector<std::vector<glm::vec4> >& points,
    const std::vector<std::vector<uint> >& indices
);

#endif //ORCINUS_FORMFACTORS_H
