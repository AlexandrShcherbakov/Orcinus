//
// Created by alex on 25.11.17.
//

#include "formfactors.h"

#include <algorithm>
#include <iostream>
#include <random>

#include <embree3/rtcore.h>


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
    RTCGeometry Geometry;
    RTCBuffer IndicesBuffer, PointsBuffer;
    const std::vector<Quad>& Quads;
    std::vector<std::vector<float> > FF;
    RTCIntersectContext IntersectionContext;
public:
    EmbreeFFJob(
        const std::vector<Quad>& quads,
        const std::vector<glm::vec4>& points,
        const std::vector<uint>& indices
    ): Quads(quads) {
        Device = rtcNewDevice(""); CHECK_EMBREE
        Scene = rtcNewScene(Device); CHECK_EMBREE
        Geometry = rtcNewGeometry(Device, RTC_GEOMETRY_TYPE_TRIANGLE); CHECK_EMBREE
        IndicesBuffer = rtcNewSharedBuffer(
            Device,
            reinterpret_cast<void*>(const_cast<unsigned *>(indices.data())),
            indices.size() * sizeof(indices[0])); CHECK_EMBREE
        PointsBuffer = rtcNewSharedBuffer(
            Device,
            reinterpret_cast<void*>(const_cast<glm::vec4*>(points.data())),
            points.size() * sizeof(points[0])); CHECK_EMBREE

        rtcSetGeometryBuffer(Geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, IndicesBuffer, 0, 0, indices.size() / 3); CHECK_EMBREE
        rtcSetGeometryBuffer(Geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, PointsBuffer, 0, sizeof(float), points.size()); CHECK_EMBREE
        rtcCommitGeometry(Geometry); CHECK_EMBREE
        rtcAttachGeometry(Scene, Geometry); CHECK_EMBREE
        rtcCommitScene(Scene); CHECK_EMBREE
        rtcInitIntersectContext(&IntersectionContext); CHECK_EMBREE
        IntersectionContext.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;
        IntersectionContext.instID[0] = 0;
    }

    ~EmbreeFFJob() {
        rtcReleaseBuffer(PointsBuffer);
        rtcReleaseBuffer(IndicesBuffer);
        rtcReleaseGeometry(Geometry);
        rtcReleaseScene(Scene);
        rtcReleaseDevice(Device);
    }

    std::vector<std::vector<float> > Execute() {
        FF.resize(Quads.size());
        for (auto &ff : FF) {
            ff.assign(Quads.size(), 0);
        }
        const uint PACKET_SIZE = 16;
        auto samples = GenerateRandomSamples(16);
        for (uint i = 0; i < FF.size(); ++i) {
            for (uint j = i + 1; j < FF[i].size(); ++j) {
                std::vector<std::pair<glm::vec4, glm::vec4> > rays;
                for (uint k = 0; k < samples.size(); ++k) {
                    const auto sample1 = Quads[i].GetSample(samples[k]);
                    for (auto sample : samples) {
                        rays.emplace_back(std::make_pair(sample1, Quads[j].GetSample(sample) - sample1));
                    }
                }
                uint visibilityCount = 0;
                float samplesSum = 0;
                const float BIAS = 1e-5f;
                std::vector<RTCRay16> raysPackets(rays.size() / PACKET_SIZE);
                for (uint k = 0; k < rays.size(); k += PACKET_SIZE) {
                    for (uint l = 0; l < PACKET_SIZE; ++l) {
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

                for (uint k = 0; k < rays.size(); k += PACKET_SIZE) {
                    const int validMask = ~1u;
                    rtcOccluded16(&validMask, Scene, &IntersectionContext, &raysPackets[k / PACKET_SIZE]); CHECK_EMBREE
                }

                for (uint k = 0; k < rays.size(); k += PACKET_SIZE) {
                    for (uint l = 0; l < PACKET_SIZE; ++l) {
                        if (std::isinf(raysPackets[k / PACKET_SIZE].tfar[l])) {
                            continue;
                        }
                        const float rayLength = glm::length(rays[k + l].second);
                        const float cosTheta1 = std::max(glm::dot(Quads[i].GetNormal(), glm::normalize(rays[k + l].second)), 0.0f);
                        const float cosTheta2 = std::max(glm::dot(Quads[j].GetNormal(), -glm::normalize(rays[k + l].second)), 0.0f);
                        const float sampleValue = cosTheta1 * cosTheta2 / sqr(rayLength);
                        if (sampleValue < 0.5 * sqr(samples.size()) * static_cast<float>(M_PI) * Quads[i].GetSquare()) {
                            visibilityCount++;
                            samplesSum += sampleValue;
                        }
                    }
                }

                if (visibilityCount) {
                    FF[i][j] = FF[j][i] = samplesSum / visibilityCount / static_cast<float>(M_PI) * Quads[i].GetSquare();
                }
            }
        }
        return FF;
    }
};

std::vector<std::vector<float> > ComputeFormFactorsEmbree(
    const std::vector<Quad>& quads,
    const std::vector<glm::vec4>& points,
    const std::vector<uint>& indices
) {
    EmbreeFFJob job(quads, points, indices);
    return job.Execute();
}