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
    std::vector<glm::vec4> BDiffuse(size);
    for (int i = 0; i < size; ++i) {
        B_sum[i].assign(size, glm::vec4(0));
        B_last[i].assign(size, glm::vec4(0));
    }
    const auto specularTensor = rho.GetSpecularSparseTensor();
    for (int it = 0; it < iter; ++it) {
        const int THREADS_COUNT = 10;

        BDiffuse.assign(size, glm::vec4(0));
#pragma omp parallel for num_threads(THREADS_COUNT)
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                BDiffuse[i] += rho.GetDiffuse(i) * ff[i][j] * B_current[j][i];
            }
        }

#pragma omp parallel for num_threads(THREADS_COUNT)
        for (int i = 0; i < size; ++i) {
            for (int k = 0; k < size; ++k) {
                B_last[i][k] += BDiffuse[i];
            }
        }

#pragma omp parallel for num_threads(THREADS_COUNT)
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                if (it > 0) {
                    B_sum[i][j] += B_last[i][j];
                }
            }
        }
        B_current = B_last;
#pragma omp parallel for num_threads(THREADS_COUNT)
        for (int i = 0; i < size; ++i) {
            B_last[i].assign(size, glm::vec4(0));
        }
    }
    return B_sum;
}
