//
// Created by alex on 25.11.17.
//

#include "formfactors.h"

#include <algorithm>
#include <iostream>
#include <random>


std::vector<glm::vec2> GenerateRandomSamples(const uint samplesNumber) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0, 1);
    std::vector<glm::vec2> samples(samplesNumber);
    generator.seed(0);
    for (auto & sample : samples) {
        sample = glm::vec2(distribution(generator), distribution(generator));
    }
    return samples;
}

bool IsQuadsVisible(const Quad &q1, const Quad &q2) {
    const std::array<glm::vec4, 5> points1 = {
        q1.GetVertices()[0].GetPoint(),
        q1.GetVertices()[1].GetPoint(),
        q1.GetVertices()[2].GetPoint(),
        q1.GetVertices()[3].GetPoint(),
        q1.GetSample(glm::vec2(0.5)),
    };
    const std::array<glm::vec4, 5> points2 = {
        q2.GetVertices()[0].GetPoint(),
        q2.GetVertices()[1].GetPoint(),
        q2.GetVertices()[2].GetPoint(),
        q2.GetVertices()[3].GetPoint(),
        q2.GetSample(glm::vec2(0.5)),
    };
    bool isVisible = false;
    for (auto& p1: points1) {
        for (auto& p2: points2) {
            isVisible |= (
                glm::dot(p2 - p1, q1.GetNormal()) > 0.0f
                && glm::dot(p1 - p2, q2.GetNormal()) > 0.0f
            );
        }
    }

    return isVisible;
}


float GetInfluenceForTwoSamples(
    const glm::vec4& sample1,
    const glm::vec4& sample2,
    const glm::vec4& normal1,
    const glm::vec4& normal2,
    const std::vector<Quad>& quadsToCheck
) {
    const glm::vec4 r = sample2 - sample1;
    const float length = glm::length(r);
    const float cosTheta1 = std::max(glm::dot(normal1, glm::normalize(r)), 0.0f);
    const float cosTheta2 = std::max(glm::dot(normal2, -glm::normalize(r)), 0.0f);
    const float sampleValue = cosTheta1 * cosTheta2 / sqr(length);

    if (sampleValue == 0.0f) {
        return 0;
    }
    bool hasIntersection = false;
    for (uint interQuadIdx = 0; interQuadIdx < quadsToCheck.size() && !hasIntersection; ++interQuadIdx) {
        hasIntersection = quadsToCheck[interQuadIdx].CheckIntersectionWithVector(sample1, sample2);
    }
    if (hasIntersection) {
        return 0;
    }
    return sampleValue;
}


inline float ComputeFormFactorForTwoQuads(
    const Quad& q1,
    const Quad& q2,
    const std::vector<Quad>& quadsToCheck
) {
    static const uint samplesNum = 20;
    static const auto samples = GenerateRandomSamples(samplesNum);

    float result = 0;
    int count = 0;
    const int th_num = samplesNum / 2;
    std::array<float, th_num> res_ar{};
    std::array<int, th_num> cnt_ar{};
    res_ar.fill(0);
    cnt_ar.fill(0);
    for (uint samplerIdx1 = 0; samplerIdx1 < samplesNum; ++samplerIdx1) {
        const glm::vec4 sample1 = q1.GetSample(samples[samplerIdx1]);
#pragma omp parallel for num_threads(th_num)
        for (uint th_i = 0; th_i < th_num; ++th_i) {
            const auto begin = static_cast<uint>(samples.size() / th_num * th_i);
            const auto end = static_cast<uint>(std::min(samples.size() / th_num * (th_i + 1), samples.size()));
            for (uint samplerIdx2 = begin; samplerIdx2 < end; ++samplerIdx2) {
                const glm::vec4 sample2 = q2.GetSample(samples[samplerIdx2]);
                const float sampleValue = GetInfluenceForTwoSamples(
                    sample1,
                    sample2,
                    q1.GetNormal(),
                    q2.GetNormal(),
                    quadsToCheck);

                if (sampleValue > 0.5 * sqr(samples.size()) * static_cast<float>(M_PI) * q1.GetSquare()) {
                    continue;
                }
                res_ar[th_i] += sampleValue;
                cnt_ar[th_i]++;
            }
        }
    }
    for (uint th_i = 0; th_i < th_num; ++th_i) {
        result += res_ar[th_i];
        count += cnt_ar[th_i];
    }

    if (count) {
        result /= count;
    }
    return result / static_cast<float>(M_PI) * q1.GetSquare();
}


std::vector<Quad> FilterQuadsForChecking(const uint quadIdx1, const uint quadIdx2, const std::vector<Quad>& quads) {
    const glm::vec4 vecBegin = quads[quadIdx1].GetSample(glm::vec2(0.5, 0.5));
    const glm::vec4 line4 = quads[quadIdx2].GetSample(glm::vec2(0.5, 0.5)) - vecBegin;
    const glm::vec3 line = {line4.x, line4.y, line4.z};
    const float lineLength = glm::length(line);
    const float thresholdDist = quads[0].GetSide() * std::sqrt(2.f);

    std::vector<Quad> result;
    for (uint i = 0; i < quads.size(); ++i) {
        if (i == quadIdx1 || i == quadIdx2) {
            continue;
        }
        const glm::vec4 sampleToQuad4 = quads[i].GetSample(glm::vec2(0.5, 0.5)) - vecBegin;
        const glm::vec3 sampleToQuad = {sampleToQuad4.x, sampleToQuad4.y, sampleToQuad4.z};
        if (glm::length(glm::cross(sampleToQuad, line)) / lineLength <= thresholdDist) {
            result.push_back(quads[i]);
        }
    }
    return result;
}


std::vector<std::vector<float> > ComputeFormFactors(
    const std::vector<Quad>& quads,
    const bool verbose,
    const bool checkIntersections,
    const bool antiradiance
) {
    static auto timestamp = time(nullptr);
    std::vector<std::vector<float> > formFactors(quads.size());
    for (auto &formFactor : formFactors) {
        formFactor.assign(quads.size(), 0);
    }
    for (uint quadIdx1 = 0; quadIdx1 < quads.size(); ++quadIdx1) {
        std::vector<glm::vec4> samplesFirst;
        uint beginQuad2 = antiradiance ? 0 : quadIdx1 + 1;
        auto quad1 = quads[quadIdx1];
        for (uint quadIdx2 = beginQuad2; quadIdx2 < quads.size(); ++quadIdx2) {
            if (quadIdx1 == quadIdx2) {
                continue;
            }
            auto quad2 = quads[quadIdx2];
            if (antiradiance) {
                quad2.SetNormal(-quad2.GetNormal());
            }
            if (!IsQuadsVisible(quads[quadIdx1], quad2)) {
                continue;
            }

            std::vector<Quad> quadsToCheck;
            if (checkIntersections && !antiradiance) {
                quadsToCheck = FilterQuadsForChecking(quadIdx1, quadIdx2, quads);
            }

            const float result = ComputeFormFactorForTwoQuads(quad1, quad2, quadsToCheck);
            formFactors[quadIdx1][quadIdx2] = result;
            if (!antiradiance) {
                formFactors[quadIdx2][quadIdx1] = result;
            }
        }
        if (verbose) {
            const ulong size = quads.size();
            if (quadIdx1 * 100 / size < (quadIdx1 + 1) * 100 / size) {
                std::cout << "FF process "
                          << (quadIdx1 + 1) * 100 / size
                          << "% ("
                          << time(nullptr) - timestamp
                          << " seconds)" << std::endl;
                timestamp = time(nullptr);
            }
        }
    }
    return formFactors;
}


std::vector<std::vector<float> > ComputeAlternativeFF(const std::vector<Quad>& quads) {
    auto ffTrans = ComputeFormFactors(quads, true, false, false);
    auto antiradiance = ComputeFormFactors(quads, true, false, true);
    for (uint i = 0; i < antiradiance.size(); ++i) {
        for (uint j = 0; j < antiradiance[i].size(); ++j) {
            antiradiance[i][j] *= -1;
            if (i == j) {
                antiradiance[i][j] = 1;
            }
        }
    }
    std::vector<std::vector<float> > ff(antiradiance.size());
    for (uint i = 0; i < ff.size(); ++i) {
        ff[i].assign(ff.size(), 0);
        for (uint j = 0; j < ff[i].size(); ++j) {
            for (uint k = 0; k < ffTrans.size(); ++k) {
                ff[i][j] += antiradiance[i][k] * ffTrans[k][j];
            }
        }
    }
    return ff;
}

inline float Radians(const float degrees) {
    return degrees / 180.0f * 3.14159f;
}

bool PointInLightCone(const Hors::SpotLight& light, const glm::vec4& point) {
    const auto lightDirection = glm::normalize(light.GetDirection());
    const auto lightToPoint = glm::normalize(glm::vec3(point) - light.GetPosition() / 2.0f);
    return std::cos(Radians(light.GetOuterAngle())) < glm::dot(lightDirection, lightToPoint);
}

bool QuadInLightCone(const Hors::SpotLight& light, const Quad& quad) {
    for (const auto & vertex: quad.GetVertices()) {
        if (PointInLightCone(light, vertex.GetPoint())) {
            return true;
        }
    }
    return false;
}

std::vector<std::vector<glm::vec4> > ComputeInitialLight(const std::vector<Quad>& quads, const Hors::SpotLight& light) {
    std::vector<std::vector<glm::vec4> > initialLight(quads.size());
    for (auto & initLine : initialLight) {
        initLine.assign(quads.size(), glm::vec4(0));
    }
    const auto samples = GenerateRandomSamples(20);
    for (uint i = 0; i < quads.size(); ++i) {
        if (!QuadInLightCone(light, quads[i])) {
            continue;
        }
        for (uint j = 0; j < samples.size(); ++j) {
            const auto sample = quads[i].GetSample(samples[j]);
            if (!PointInLightCone(light, sample)) {
                continue;
            }
            if (glm::dot(quads[i].GetNormal(), glm::vec4(light.GetPosition(), 1.0f) - sample) < 1e-9f) {
                continue;
            }
            int nearestHitQuad = -1;
            float hitCoef = 0;
            for (uint k = 0; k < quads.size(); ++k) {
                if (k == i) {
                    continue;
                }
                const float t = quads[k].GetHitVectorCoef(sample, glm::vec4(light.GetPosition(), 1));
                if (std::isnan(t) || t <= 0) {
                    continue;
                }
                const auto virtualEndPoint = sample + (glm::vec4(light.GetPosition(), 1) - sample) * t * 2.0f;
                if (!quads[k].CheckIntersectionWithVector(sample, virtualEndPoint)) {
                    continue;
                }
//                std::cout << t << std::endl;
                if (t <= 1) {
                    nearestHitQuad = -1;
                    break;
                }
                if (nearestHitQuad == -1 || hitCoef > t) {
                    hitCoef = t;
                    nearestHitQuad = k;
                }
            }
            if (nearestHitQuad != -1) {
                initialLight[nearestHitQuad][i] += glm::vec4(light.GetColor(), 1)
                    / static_cast<float>(samples.size())
                    / quads[i].GetSquare()
                    * light.GetMultiplier();// * 150.f;
//                    * glm::dot(quads[i].GetNormal(), glm::normalize(glm::vec4(light.GetPosition(), 1) - sample));
            }
        }
    }
    return initialLight;
}
