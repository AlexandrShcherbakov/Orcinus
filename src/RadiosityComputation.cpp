//
// Created by alex on 20.01.18.
//

#include <RadiosityComputation.h>

#include <iostream>


std::vector<glm::vec4> RecomputeColorsForQuadsCPU(
    const std::vector<std::vector<float> > & ff,
    const std::vector<glm::vec4> & colors
) {
    std::vector<glm::vec4> lighting(colors.size());
    for (uint i = 0; i < lighting.size(); ++i) {
        for (uint j = 0; j < colors.size(); ++j) {
            lighting[i] += colors[j] * ff[i][j];
        }
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


std::vector<std::vector<glm::vec4> > ComputeRadiosityCPU(
    const VirtualBRDFTensor& rho,
    const std::vector<std::vector<float> >& ff,
    const std::vector<std::vector<glm::vec4> >& initLight,
    const int iter
) {
    auto B_last = initLight;
    auto B_current = B_last;
    const int size = B_last.size();
    std::vector<std::vector<glm::vec4> > B_sum(size);
    for (int i = 0; i < size; ++i) {
        B_sum[i].assign(size, glm::vec4(0));
        B_last[i].assign(size, glm::vec4(0));
    }
//    for (int i = 0; i < size; ++i) {
//        for (int j = 0; j < size; ++j) {
//            B_sum[i][j] = initLight[i][j];
//        }
//    }
    for (int it = 0; it < iter; ++it) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                if (!ff[i][j] || !glm::length(B_current[j][i])) {
                    continue;
                }
                for (int k = 0; k < size; ++k) {
                    B_last[i][k] += glm::vec4(rho.GetValue(i, j, k), 1) * ff[i][j] * B_current[j][i];
//                    std::cout << B_last[i][k].x << ' ' << B_last[i][k].y << ' ' << B_last[i][k].z << std::endl;
                }
            }
        }
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                B_sum[i][j] = B_last[i][j];
            }
        }
        B_current = B_last;
        for (int i = 0; i < size; ++i) {
            B_last[i].assign(size, glm::vec4(0));
        }
    }
    return B_sum;
}
