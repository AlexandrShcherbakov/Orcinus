//
// Created by alex on 20.01.18.
//

#include <RadiosityComputation.h>

#include <algorithm>
#include <iostream>
#include <set>
#include <utility>


std::vector<glm::vec4> RecomputeColorsForQuadsCPU(
    const std::vector<std::vector<float> > & ff,
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
            for (uint j = 0; j < colors.size(); ++j) {
                bounce[i] += prevBounce[j] * ff[i][j];
            }
        }
        for (uint i = 0; i < lighting.size(); ++i) {
            lighting[i] += bounce[i] * colors[i];
            prevBounce[i] = bounce[i];
            bounce[i] = glm::vec4(0);
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

uint FindParent(std::vector<uint>& parents, const uint v) {
    if (parents[v] == v) {
        return v;
    }
    return parents[v] = FindParent(parents, parents[v]);
}

bool UnionSets(std::vector<uint>& parents, std::vector<uint>& rang, const uint a, const uint b) {
    auto parentA = FindParent(parents, a);
    auto parentB = FindParent(parents, b);
    if (parentA != parentB) {
        if (rang[parentA] < rang[parentB]) {
            std::swap(parentA, parentB);
        }
        parents[parentB] = parentA;
        if (rang[parentA] == rang[parentB]) {
            rang[parentB]++;
        }
        return true;
    }
    return false;
}

bool UnionSets(std::vector<uint>& parents, std::vector<uint>& rang, std::vector<uint>& sizes, const uint a, const uint b) {
    auto parentA = FindParent(parents, a);
    auto parentB = FindParent(parents, b);
    if (parentA != parentB) {
        if (rang[parentA] < rang[parentB]) {
            std::swap(parentA, parentB);
        }
        sizes[parentA] += sizes[parentB];
        parents[parentB] = parentA;
        if (rang[parentA] == rang[parentB]) {
            rang[parentB]++;
        }
        return true;
    }
    return false;
}

std::vector<std::vector<uint> > HierarchicalClusterization(const std::vector<std::vector<float> >& matrix) {
    std::vector<std::tuple<float, uint, uint> > matrixValues;
    for (uint j = 0; j < matrix.size(); ++j) {
        for (uint i = j + 1; i < matrix[j].size(); ++i) {
            if (matrix[j][i]) {
                matrixValues.emplace_back(std::make_tuple(matrix[j][i], j, i));
            }
        }
    }
    std::sort(matrixValues.rbegin(), matrixValues.rend());

    std::vector<uint> parent(matrix.size());
    std::vector<uint> rang(matrix.size());
    for (uint i = 0; i < parent.size(); ++i) {
        parent[i] = i;
        rang[i] = 0;
    }
    auto clustersNumber = static_cast<uint>(parent.size());
    const auto targetClusterNumber = static_cast<uint>(std::sqrt(clustersNumber));
    for (uint i = 0; i < matrixValues.size() && clustersNumber > targetClusterNumber; ++i) {
        clustersNumber -= UnionSets(parent, rang, std::get<1>(matrixValues[i]), std::get<2>(matrixValues[i]));
    }
    decltype(matrixValues) emptyMatrixValues;
    matrixValues.swap(emptyMatrixValues);
    std::map<uint, uint> clustersRoots;
    for (uint i = 0; i < parent.size(); ++i) {
        if (clustersRoots.find(FindParent(parent, i)) == clustersRoots.end()) {
            clustersRoots[FindParent(parent, i)] = static_cast<uint>(clustersRoots.size());
        }
    }
    std::vector<std::vector<uint> > clusters(clustersNumber);
    for (uint k = 0; k < parent.size(); ++k) {
        clusters[clustersRoots[parent[k]]].push_back(k);
    }
    return clusters;
}

std::vector<std::vector<uint> > HierarchicalClusterizationSizeRestriction(const std::vector<std::vector<float> >& matrix) {
    std::vector<std::tuple<float, uint, uint> > matrixValues;
    for (uint j = 0; j < matrix.size(); ++j) {
        for (uint i = j + 1; i < matrix[j].size(); ++i) {
            if (matrix[j][i]) {
                matrixValues.emplace_back(std::make_tuple(matrix[j][i], j, i));
            }
        }
    }
    std::sort(matrixValues.rbegin(), matrixValues.rend());

    std::vector<uint> parent(matrix.size());
    std::vector<uint> rang(matrix.size());
    std::vector<uint> sizes(matrix.size());
    for (uint i = 0; i < parent.size(); ++i) {
        parent[i] = i;
        rang[i] = 0;
        sizes[i] = 1;
    }
    auto clustersNumber = static_cast<uint>(parent.size());
    const auto targetClusterNumber = static_cast<uint>(std::sqrt(clustersNumber));
    const auto maxClusterSize = (clustersNumber + targetClusterNumber - 1) / targetClusterNumber;
    for (uint i = 0; i < matrixValues.size() && clustersNumber > targetClusterNumber; ++i) {
        if (sizes[FindParent(parent, std::get<1>(matrixValues[i]))]
            + sizes[FindParent(parent, std::get<2>(matrixValues[i]))] > maxClusterSize) {
            continue;
        }
        clustersNumber -= UnionSets(parent, rang, sizes, std::get<1>(matrixValues[i]), std::get<2>(matrixValues[i]));
    }
    decltype(matrixValues) emptyMatrixValues;
    matrixValues.swap(emptyMatrixValues);
    std::map<uint, uint> clustersRoots;
    for (uint i = 0; i < parent.size(); ++i) {
        if (clustersRoots.find(FindParent(parent, i)) == clustersRoots.end()) {
            clustersRoots[FindParent(parent, i)] = static_cast<uint>(clustersRoots.size());
        }
    }
    std::vector<std::vector<uint> > clusters(clustersNumber);
    for (uint k = 0; k < parent.size(); ++k) {
        clusters[clustersRoots[parent[k]]].push_back(k);
    }
    return clusters;
}


std::vector<std::vector<float> > SimplifyMatrixUsingClasters(
    std::vector<std::vector<float> > ff,
    const std::vector<std::vector<uint> >& clusters
) {
    for (uint i = 0; i < clusters.size(); ++i) {
        for (uint j = i + 1; j < clusters.size(); ++j) {
            float sum = 0;
            for (const auto& idxInFirstCluster: clusters[i]) {
                for (const auto& idxInSecondCluster: clusters[j]) {
                    sum += ff[idxInFirstCluster][idxInSecondCluster] + ff[idxInSecondCluster][idxInFirstCluster];
                }
            }
            const float meanValue = sum / clusters[i].size() / clusters[j].size() / 2;
            for (const auto& idxInFirstCluster: clusters[i]) {
                for (const auto& idxInSecondCluster: clusters[j]) {
                    ff[idxInFirstCluster][idxInSecondCluster] = ff[idxInSecondCluster][idxInFirstCluster] = meanValue;
                }
            }
        }
    }
    return ff;
}

static void normalize(std::vector<float>& row) {
    float sum = 0;
    for (const auto value: row) {
        sum += value;
    }
    for (auto& value: row) {
        value /= sum;
    }
}

static void expandMatrix(std::vector<std::vector<float> >& matrix, const int power) {
    std::vector<std::vector<float> > initMatrix(matrix);
    for (uint i = 0; i < initMatrix.size(); ++i) {
        for (uint j = i + 1; j < initMatrix[i].size(); ++j) {
            std::swap(initMatrix[i][j], initMatrix[j][i]);
        }
    }
    std::vector<std::vector<float> > resultMatrix(matrix);
    const auto size = static_cast<int>(matrix.size());
    for (int i = 0; i < power; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                float value = 0;
                for (int l = 0; l < size; ++l) {
                    value += matrix[j][l] * initMatrix[k][l];
                }
                resultMatrix[j][k] = value;
            }
        }
        matrix = resultMatrix;
    }
}

static void inflate(std::vector<float>& row, const float power) {
    for (auto& value: row) {
        value = std::pow(value, power);
    }
    normalize(row);
}

bool clustered(const std::vector<std::vector<float> >& matrix) {
    for (uint i = 0; i < matrix.size(); ++i) {
        float columnValue = 0;
        uint j = 0;
        for (; j < matrix[i].size(); ++j) {
            if (matrix[j][i]) {
                columnValue = matrix[j][i];
            }
        }
        for (; j < matrix.size(); ++j) {
            if (matrix[j][i] && std::abs(matrix[j][i] - columnValue) > 1e-8) {
                return false;
            }
        }
    }
    return true;
}

std::vector<std::vector<uint> > MCL(std::vector<std::vector<float> > matrix) {
    for (uint i = 0; i < matrix.size(); ++i) {
        matrix[i][i] = 1;
    }
    for (auto& row: matrix) {
        normalize(row);
    }
    for (int i = 0; i < 100 && !clustered(matrix); ++i) {
        expandMatrix(matrix, 2);
        for (auto& row: matrix) {
            inflate(row, 2);
        }
    }

    for (uint i = 0; i < matrix.size(); ++i) {
        for (uint j = i + 1; j < matrix[i].size(); ++j) {
            std::swap(matrix[i][j], matrix[j][i]);
        }
    }

    std::set<uint> used;
    std::vector<std::vector<uint> > clusters;
    for (uint i = 0; i < matrix.size(); ++i) {
        std::vector<uint> cluster;
        for (uint j = 0; j < matrix[i].size(); ++j) {
            if (matrix[i][j] > 1e-9) {
                if (used.count(j)) {
                    break;
                }
                cluster.push_back(j);
                used.insert(j);
            }
        }
        if (!cluster.empty()) {
            clusters.push_back(cluster);
        }
    }
    return clusters;
}
