//
// Created by alex on 20.01.18.
//

#include <RadiosityComputation.h>

#include <algorithm>
#include <iostream>
#include <set>
#include <unordered_set>
#include <utility>


std::vector<glm::vec4> RecomputeColorsForQuadsCPU(
    const std::vector<std::map<uint, float> > & ff,
    const std::vector<glm::vec4> & colors,
    const std::vector<glm::vec4> & emission,
    const int iters
) {
    const int THREADS_COUNT = 10;
    std::vector<glm::vec4> lighting = emission;
    std::vector<glm::vec4> bounce(colors.size());
    std::vector<glm::vec4> prevBounce = emission;
    for (int k = 0; k < iters; ++k) {
#pragma omp parallel for num_threads(THREADS_COUNT)
        for (uint i = 0; i < lighting.size(); ++i) {
            for (const auto& ffItem: ff[i]) {
                bounce[i] += prevBounce[ffItem.first] * ffItem.second;
            }
        }
        for (uint i = 0; i < lighting.size(); ++i) {
            lighting[i] += bounce[i] * colors[i];
            prevBounce[i] = bounce[i] * colors[i];
            bounce[i] = glm::vec4(0);
        }
    }
    return lighting;
}

void RemoveUnnecessaryQuads(
    QuadsContainer& quads,
    std::vector<std::map<int, float> > & ff
) {
    int z = 0;
    for (int i = quads.GetSize() - 1; i >= 0; --i) {
        z += ff[i].empty() && quads.GetChildren(i).first < 0;
        if (ff[i].empty() && quads.GetChildren(i).first < 0) {
            for (int j = 0; j < quads.GetSize(); ++j) {
                if (!quads.HasChildren(j)) {
                    continue;
                }
                auto children = quads.GetChildren(j);
                if (children.first > i) {
                    --children.first;
                }
                if (children.second > i) {
                    --children.second;
                }
                quads.SetChildren(j, children);
            }
            for (auto &j : ff) {
                std::map<int, float> updated;
                for (const auto it : j) {
                    if (it.first > i) {
                        updated[it.first - 1] = it.second;
                    }
                }
                for (const auto it : updated) {
                    j.erase(it.first + 1);
                }
            }
            ff.erase(ff.begin() + i);
        }
    }
    std::cout << z << std::endl;
}
