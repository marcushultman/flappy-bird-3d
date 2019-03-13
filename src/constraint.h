#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

namespace flappy {

using Indices = std::vector<uint32_t>;
using Positions = std::vector<glm::vec3>;
using Solve =
    std::function<void(uint32_t iterations, Positions &positions, Positions &rest_positions)>;

using Constraint = std::pair<Indices, Solve>;

Constraint makeCollisionConstraint(uint32_t i, const glm::vec3 &target) {
  auto solve = [i, target](auto, auto &positions, auto &) { positions[i] = target; };
  return {{i}, solve};
};

Constraint makeDistanceConstraint(std::array<uint32_t, 2> indices, std::vector<float> &weights) {
  std::sort(indices.begin(), indices.end());
  auto stiffness = (weights[indices[0]] + weights[indices[1]]) / 2.0f;
  auto solve = [indices, stiffness](auto iterations, auto &positions, auto &rest_positions) {
    auto &[i0, i1] = indices;

    auto diff = positions[i1] - positions[i0];
    auto length = glm::length(diff);
    auto normal = diff / length;

    auto coeffecient = 1 - pow(1 - stiffness, 1 / static_cast<float>(iterations));
    auto rest_length = glm::length(rest_positions[i1] - rest_positions[i0]);
    auto strain = coeffecient * normal * (length - rest_length) / 2.0f;

    positions[i0] += strain;
    positions[i1] -= strain;
  };
  return {{indices.begin(), indices.end()}, solve};
}

Constraint makeBendingConstraint(std::array<uint32_t, 4> indices, std::vector<float> &weights) {
  std::sort(indices.begin(), indices.begin() + 2);
  std::sort(indices.begin() + 2, indices.end());
  auto stiffness =
      (weights[indices[0]] + weights[indices[1]] + weights[indices[2]] + weights[indices[3]]) / 4;
  auto solve = [indices, stiffness](auto iterations, auto &positions, auto &rest_positions) {
    auto &[i0, i1, i2, i3] = indices;

    glm::vec3 p1 = positions[i1] - positions[i0];
    glm::vec3 p2 = positions[i2] - positions[i0];
    glm::vec3 p3 = positions[i3] - positions[i0];

    glm::vec3 cp1p2 = glm::cross(p1, p2);
    glm::vec3 cp1p3 = glm::cross(p1, p3);

    glm::vec3 n1 = glm::normalize(cp1p2);
    glm::vec3 n2 = glm::normalize(cp1p3);
    float d = std::max(0.0f, std::min(glm::dot(n1, n2), 1.0f));

    glm::vec3 q3 = (glm::cross(p1, n2) + glm::cross(n1, p1) * d) / glm::length(cp1p2);
    glm::vec3 q4 = (glm::cross(p1, n1) + glm::cross(n2, p1) * d) / glm::length(cp1p3);
    glm::vec3 q2 = -(glm::cross(p2, n2) + glm::cross(n1, p2) * d) / glm::length(cp1p2) -
                   (glm::cross(p3, n1) + glm::cross(n2, p3) * d) / glm::length(cp1p3);
    glm::vec3 q1 = -q2 - q3 - q4;

    glm::vec3 s2 = rest_positions[i1] - rest_positions[i0];
    glm::vec3 s3 = rest_positions[i2] - rest_positions[i0];
    glm::vec3 s4 = rest_positions[i3] - rest_positions[i0];

    glm::vec3 sn1 = glm::normalize(glm::cross(s2, s3));
    glm::vec3 sn2 = glm::normalize(glm::cross(s2, s4));
    float rd = std::max(0.0f, std::min(glm::dot(sn1, sn2), 1.0f));

    float sum = glm::length2(q1) + glm::length2(q2) + glm::length2(q3) + glm::length2(q4);
    float coe = sum != 0 ? coe = -glm::sqrt(1 - d * d) * (acos(d) - acos(rd)) / sum : 0;

    float k = 1 - pow(1 - stiffness, 1 / static_cast<float>(iterations));
    coe *= k;

    positions[i0] += coe * q1;
    positions[i1] += coe * q2;
    positions[i2] += coe * q3;
    positions[i3] += coe * q4;
  };
  return {{indices.begin(), indices.end()}, solve};
}

}  // namespace flappy
