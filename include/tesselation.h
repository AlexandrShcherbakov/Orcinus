//
// Created by alex on 24.11.17.
//

#ifndef ORCINUS_TESSELATION_H
#define ORCINUS_TESSELATION_H

#include <string>
#include <vector>

#include "GeometryTypes.h"

std::vector<Quad> ExtractQuadsFromScene(const std::vector<HydraGeomData>& meshes);

std::vector<Quad> ExtractQuadsFromScene(const HydraGeomData& data);

std::vector<Quad> TessellateScene(const std::vector<Quad> &quads, const float MinCellWidth);

void SaveTessellation(const std::vector<Quad> &quads, const std::string &path);

#endif //ORCINUS_TESSELATION_H
