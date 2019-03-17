//
// Created by alex on 25.11.17.
//

#include "formfactors.h"

#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>

#include <embree3/rtcore.h>


std::vector<glm::vec2> GenerateRandomSamples(const unsigned samplesNumber) {
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
    for (unsigned interQuadIdx = 0; interQuadIdx < quadsToCheck.size() && !hasIntersection; ++interQuadIdx) {
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
    static const unsigned samplesNum = 20;
    static const auto samples = GenerateRandomSamples(samplesNum);

    float result = 0;
    int count = 0;
    const int th_num = samplesNum / 2;
    std::array<float, th_num> res_ar{};
    std::array<int, th_num> cnt_ar{};
    res_ar.fill(0);
    cnt_ar.fill(0);
    for (unsigned samplerIdx1 = 0; samplerIdx1 < samplesNum; ++samplerIdx1) {
        const glm::vec4 sample1 = q1.GetSample(samples[samplerIdx1]);
#pragma omp parallel for num_threads(th_num)
        for (unsigned th_i = 0; th_i < th_num; ++th_i) {
            const auto begin = static_cast<unsigned>(samples.size() / th_num * th_i);
            const auto end = static_cast<unsigned>(std::min(samples.size() / th_num * (th_i + 1), samples.size()));
            for (unsigned samplerIdx2 = begin; samplerIdx2 < end; ++samplerIdx2) {
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
    for (unsigned th_i = 0; th_i < th_num; ++th_i) {
        result += res_ar[th_i];
        count += cnt_ar[th_i];
    }

    if (count) {
        result /= count;
    }
    return result / static_cast<float>(M_PI) * q1.GetSquare();
}


std::vector<Quad> FilterQuadsForChecking(const unsigned quadIdx1, const unsigned quadIdx2, const std::vector<Quad>& quads) {
    const glm::vec4 vecBegin = quads[quadIdx1].GetSample(glm::vec2(0.5, 0.5));
    const glm::vec4 line4 = quads[quadIdx2].GetSample(glm::vec2(0.5, 0.5)) - vecBegin;
    const glm::vec3 line = {line4.x, line4.y, line4.z};
    const float lineLength = glm::length(line);
    const float thresholdDist = quads[0].GetMaxSide() * std::sqrt(2.f);

    std::vector<Quad> result;
    for (unsigned i = 0; i < quads.size(); ++i) {
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
    for (unsigned quadIdx1 = 0; quadIdx1 < quads.size(); ++quadIdx1) {
        std::vector<glm::vec4> samplesFirst;
        unsigned beginQuad2 = antiradiance ? 0 : quadIdx1 + 1;
        auto quad1 = quads[quadIdx1];
        for (unsigned quadIdx2 = beginQuad2; quadIdx2 < quads.size(); ++quadIdx2) {
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
            const long long size = quads.size();
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


inline float Radians(const float degrees) {
    return degrees / 180.0f * 3.14159f;
}


#define CHECK_EMBREE \
{ \
    int errCode; \
    if ((errCode = rtcGetDeviceError(Device)) != RTC_ERROR_NONE) {\
        std::cerr << "Embree error: " << errCode  << " line: " << __LINE__ << std::endl; \
        exit(1); \
    }\
}

class EmbreeFFJob {
    RTCDevice Device;
    RTCScene Scene;
    const std::vector<Quad>& Quads;
    std::vector<std::map<unsigned, float> > FF;
    RTCIntersectContext IntersectionContext;
public:
    EmbreeFFJob(
        const std::vector<Quad>& quads,
        const std::vector<std::vector<glm::vec4> >& points,
        const std::vector<std::vector<unsigned> >& indices
    ): Quads(quads) {
        Device = rtcNewDevice(""); CHECK_EMBREE
        Scene = rtcNewScene(Device); CHECK_EMBREE
        for (unsigned i = 0; i < points.size(); ++i) {
            RTCGeometry geometry = rtcNewGeometry(Device, RTC_GEOMETRY_TYPE_TRIANGLE); CHECK_EMBREE
            RTCBuffer indicesBuffer = rtcNewSharedBuffer(
                Device,
                reinterpret_cast<void*>(const_cast<unsigned *>(indices[i].data())),
                indices[i].size() * sizeof(indices[i][0])); CHECK_EMBREE
            RTCBuffer pointsBuffer = rtcNewSharedBuffer(
                Device,
                reinterpret_cast<void*>(const_cast<glm::vec4*>(points[i].data())),
                points[i].size() * sizeof(points[i][0])); CHECK_EMBREE

            rtcSetGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, indicesBuffer, 0, 0, indices[i].size() / 3); CHECK_EMBREE
            rtcSetGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, pointsBuffer, 0, sizeof(float), points[i].size()); CHECK_EMBREE
            rtcCommitGeometry(geometry); CHECK_EMBREE
            rtcAttachGeometry(Scene, geometry); CHECK_EMBREE
            rtcReleaseGeometry(geometry);
        }
        rtcCommitScene(Scene); CHECK_EMBREE
        rtcInitIntersectContext(&IntersectionContext); CHECK_EMBREE
        IntersectionContext.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;
        IntersectionContext.instID[0] = 0;
    }

    ~EmbreeFFJob() {
        //TODO: release buffers
//        rtcReleaseBuffer(PointsBuffer);
//        rtcReleaseBuffer(IndicesBuffer);
        rtcReleaseScene(Scene);
        rtcReleaseDevice(Device);
    }

    std::vector<std::map<unsigned, float> > Execute() {
        FF.resize(Quads.size());
        const unsigned PACKET_SIZE = 16;
        const auto samples = GenerateRandomSamples(16);
        for (unsigned i = 0; i < FF.size(); ++i) {
            for (unsigned j = i + 1; j < FF.size(); ++j) {
                std::vector<std::pair<glm::vec4, glm::vec4> > rays;
                for (const auto firstQuadSample : samples) {
                    const auto sample1 = Quads[i].GetSample(firstQuadSample);
                    for (auto secondQuadSample : samples) {
                        rays.emplace_back(std::make_pair(sample1, Quads[j].GetSample(secondQuadSample) - sample1));
                    }
                }
                unsigned visibilityCount = 0;
                float samplesSum = 0;
                const float BIAS = 1e-5f;
                std::vector<RTCRay16> raysPackets(rays.size() / PACKET_SIZE);
                for (unsigned k = 0; k < rays.size(); k += PACKET_SIZE) {
                    for (unsigned l = 0; l < PACKET_SIZE; ++l) {
                        raysPackets[k / PACKET_SIZE].org_x[l] = rays[k + l].first.x;
                        raysPackets[k / PACKET_SIZE].org_y[l] = rays[k + l].first.y;
                        raysPackets[k / PACKET_SIZE].org_z[l] = rays[k + l].first.z;
                        raysPackets[k / PACKET_SIZE].tnear[l] = BIAS;
                        raysPackets[k / PACKET_SIZE].dir_x[l] = rays[k + l].second.x;
                        raysPackets[k / PACKET_SIZE].dir_y[l] = rays[k + l].second.y;
                        raysPackets[k / PACKET_SIZE].dir_z[l] = rays[k + l].second.z;
                        raysPackets[k / PACKET_SIZE].tfar[l] = glm::length(rays[k + l].second) - BIAS;
                        raysPackets[k / PACKET_SIZE].id[l] = 0;
                        raysPackets[k / PACKET_SIZE].mask[l] = 0;
                        raysPackets[k / PACKET_SIZE].time[l] = 0;
                    }
                }

                for (unsigned k = 0; k < rays.size(); k += PACKET_SIZE) {
                    const int validMask = ~0u;
                    rtcOccluded16(&validMask, Scene, &IntersectionContext, &raysPackets[k / PACKET_SIZE]); CHECK_EMBREE
                }

                for (unsigned k = 0; k < rays.size(); k += PACKET_SIZE) {
                    for (unsigned l = 0; l < PACKET_SIZE; ++l) {
                        if (std::isinf(raysPackets[k / PACKET_SIZE].tfar[l])) {
                            continue;
                        }
                        const float rayLength = glm::length(rays[k + l].second);
                        const float cosTheta1 = std::max(glm::dot(Quads[i].GetNormal(), glm::normalize(rays[k + l].second)), 0.0f);
                        const float cosTheta2 = std::max(glm::dot(Quads[j].GetNormal(), -glm::normalize(rays[k + l].second)), 0.0f);
                        const float sampleValue = cosTheta1 * cosTheta2 / sqr(rayLength);
                        if (sampleValue < 0.5 * sqr(samples.size()) * static_cast<float>(M_PI)) {
                            visibilityCount++;
                            samplesSum += sampleValue;
                        }
                    }
                }

                if (visibilityCount && samplesSum) {
                    float value = samplesSum / visibilityCount / static_cast<float>(M_PI) * Quads[i].GetSquare() * Quads[j].GetSquare();
                    FF[i][j] = value / Quads[i].GetSquare();
                    FF[j][i] = value / Quads[j].GetSquare();
                }
            }
        }
        return FF;
    }
};

std::vector<std::map<unsigned, float> > ComputeFormFactorsEmbree(
    const std::vector<Quad>& quads,
    const std::vector<std::vector<glm::vec4> >& points,
    const std::vector<std::vector<unsigned> >& indices
) {
    EmbreeFFJob job(quads, points, indices);
    return job.Execute();
}

class EmbreeHierarchicalFFJob {
    RTCDevice Device;
    RTCScene Scene;
    QuadsContainer& Quads;
    std::vector<std::map<int, float> > FF;
    RTCIntersectContext IntersectionContext;
    const float EPS = 1e-7;
    std::map<std::pair<int, int>, float> cache;
    std::vector<Quad> bigQuads;

    float GetTwoQuadsFF(const int idx1, const int idx2) {
        const glm::vec4 normal1 = Quads.GetQuad(idx1).GetNormal();
        const glm::vec4 normal2 = Quads.GetQuad(idx2).GetNormal();
        const glm::vec4 RR = glm::normalize(Quads.GetQuad(idx1).GetSample(glm::vec2(0.5)) - Quads.GetQuad(idx2).GetSample(glm::vec2(0.5)));
        if (glm::dot(-1.f * RR, normal1) < -1e-3 || glm::dot(RR, normal2) < -1e-3) {
            return 0;
        }

        const unsigned PACKET_SIZE = 16;
        const auto samples = GenerateRandomSamples(PACKET_SIZE * 2);

        std::vector<std::pair<glm::vec4, glm::vec4> > rays;
        for (const auto firstQuadSample : samples) {
            const auto sample1 = Quads.GetQuad(idx1).GetSample(firstQuadSample);
            for (auto secondQuadSample : samples) {
                rays.emplace_back(std::make_pair(sample1, Quads.GetQuad(idx2).GetSample(secondQuadSample) - sample1));
            }
        }
        unsigned visibilityCount = 0;
        float samplesSum = 0;
        const float BIAS = 1e-7f;
        std::vector<RTCRay16> raysPackets(rays.size() / PACKET_SIZE);
        for (unsigned k = 0; k < rays.size(); k += PACKET_SIZE) {
            for (unsigned l = 0; l < PACKET_SIZE; ++l) {
                raysPackets[k / PACKET_SIZE].org_x[l] = rays[k + l].first.x;
                raysPackets[k / PACKET_SIZE].org_y[l] = rays[k + l].first.y;
                raysPackets[k / PACKET_SIZE].org_z[l] = rays[k + l].first.z;
                raysPackets[k / PACKET_SIZE].tnear[l] = BIAS;
                raysPackets[k / PACKET_SIZE].dir_x[l] = rays[k + l].second.x;
                raysPackets[k / PACKET_SIZE].dir_y[l] = rays[k + l].second.y;
                raysPackets[k / PACKET_SIZE].dir_z[l] = rays[k + l].second.z;
                raysPackets[k / PACKET_SIZE].tfar[l] = 1 - BIAS;
                raysPackets[k / PACKET_SIZE].id[l] = 0;
                raysPackets[k / PACKET_SIZE].mask[l] = 0;
                raysPackets[k / PACKET_SIZE].time[l] = 0;
            }
        }

#pragma omp parallel for
        for (unsigned k = 0; k < rays.size(); k += PACKET_SIZE) {
            const int validMask = ~0u;
            rtcInitIntersectContext(&IntersectionContext); CHECK_EMBREE
            IntersectionContext.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;
            IntersectionContext.instID[0] = 0;
            rtcOccluded16(&validMask, Scene, &IntersectionContext, &raysPackets[k / PACKET_SIZE]); CHECK_EMBREE
        }

        for (unsigned k = 0; k < rays.size(); k += PACKET_SIZE) {
            for (unsigned l = 0; l < PACKET_SIZE; ++l) {
                if (raysPackets[k / PACKET_SIZE].tfar[l] < 0.f) {
                    continue;
                }
                const float rayLength = glm::length(rays[k + l].second);
                const float cosTheta1 = std::max(glm::dot(normal1, glm::normalize(rays[k + l].second)), 0.0f);
                const float cosTheta2 = std::max(glm::dot(normal2, -glm::normalize(rays[k + l].second)), 0.0f);
                const float sampleValue = cosTheta1 * cosTheta2 / sqr(rayLength);
                if (sampleValue < 0.5 * sqr(samples.size()) * static_cast<float>(M_PI)) {
                    samplesSum += sampleValue;
                    visibilityCount++;
                }
            }
        }

        if (visibilityCount && samplesSum) {
            return samplesSum / visibilityCount / static_cast<float>(M_PI) * Quads.GetQuad(idx1).GetSquare() * Quads.GetQuad(idx2).GetSquare();
        }
        return 0;
    }

    void ProcessTwoQuads(const int idx1, const int idx2) {
        const float ff = GetTwoQuadsFF(idx1, idx2);
        if (ff < 1e-9f) {
            return;
        }
        FF[idx1][idx2] = ff / Quads.GetQuad(idx2).GetSquare();
        FF[idx2][idx1] = ff / Quads.GetQuad(idx1).GetSquare();
    }

public:
    EmbreeHierarchicalFFJob(
        QuadsContainer& quads,
        const std::vector<std::vector<glm::vec4> >& points,
        const std::vector<std::vector<unsigned> >& indices
    ): Quads(quads) {
        Device = rtcNewDevice(""); CHECK_EMBREE
        Scene = rtcNewScene(Device); CHECK_EMBREE
        for (unsigned i = 0; i < points.size(); ++i) {
            RTCGeometry geometry = rtcNewGeometry(Device, RTC_GEOMETRY_TYPE_TRIANGLE); CHECK_EMBREE
            unsigned* ptrIdx = reinterpret_cast<unsigned*>(rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(unsigned) * 3, indices[i].size() / 3)); CHECK_EMBREE
            for (unsigned j = 0; j < indices[i].size(); ++j) {
                ptrIdx[j] = indices[i][j];
            }
            glm::vec3* ptrPnt = reinterpret_cast<glm::vec3*>(rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(glm::vec3), points[i].size())); CHECK_EMBREE
            for (unsigned j = 0; j < points[i].size(); ++j) {
                ptrPnt[j] =  glm::vec3(points[i][j]);
            }
            rtcCommitGeometry(geometry); CHECK_EMBREE
            rtcAttachGeometry(Scene, geometry); CHECK_EMBREE
            rtcReleaseGeometry(geometry);
        }
        rtcCommitScene(Scene); CHECK_EMBREE
    }

    ~EmbreeHierarchicalFFJob() {
        //TODO: release buffers
//        rtcReleaseBuffer(PointsBuffer);
//        rtcReleaseBuffer(IndicesBuffer);
        rtcReleaseScene(Scene);
        rtcReleaseDevice(Device);
    }

    std::vector<std::map<int, float>> Execute(const unsigned maxDepth) {
//        std::vector<Quad> result;
//        for (int i = 0; i < Quads.GetSize(); ++i) {
//            bigQuads.push_back(Quads.GetQuad(i));
//            const auto subdivision = Quads.GetQuad(i).Tessellate(maxDepth);
//            result.insert(result.end(), subdivision.begin(), subdivision.end());
//        }
//        const int initSizeCount = Quads.GetSize();
//        for (int i = initSizeCount - 1; i >= 0; --i) {
//            Quads.RemoveQuad(i);
//        }
//        for (const auto &i : result) {
//            Quads.AddQuad(i);
//        }
        FF.resize(static_cast<unsigned long>(Quads.GetSize()));
        unsigned sum = 0;
        for (int i = 0; i < Quads.GetSize(); ++i) {
            for (int j = i + 1; j < Quads.GetSize(); ++j) {
                ProcessTwoQuads(i, j);
            }
            sum += FF[i].size();
            std::cout << i << "-th patch processed. " << sum << " values total." << std::endl;
        }
        return FF;
    }
};

std::vector<std::map<int, float>> ComputeFormFactorsEmbree(
    QuadsContainer& quads,
    const std::vector<std::vector<glm::vec4> >& points,
    const std::vector<std::vector<unsigned> >& indices,
    const unsigned maxDepth
) {
    EmbreeHierarchicalFFJob job(quads, points, indices);
    return job.Execute(maxDepth);
}