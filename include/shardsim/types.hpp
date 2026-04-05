#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace shardsim {

struct Vec2u {
    std::size_t x {0};
    std::size_t y {0};
};

struct Vec3u {
    std::size_t x {0};
    std::size_t y {0};
    std::size_t z {0};
};

struct RuntimeMetadata {
    std::string run_id;
    std::string git_ref;
    std::string created_at_utc;
};

struct Field2D {
    Vec2u size;
    std::vector<double> values;

    [[nodiscard]] double at(std::size_t i, std::size_t j) const {
        return values.at(j * size.x + i);
    }

    double& at(std::size_t i, std::size_t j) {
        return values.at(j * size.x + i);
    }
};

struct Field3D {
    Vec3u size;
    std::vector<double> values;

    [[nodiscard]] double at(std::size_t i, std::size_t j, std::size_t k) const {
        return values.at((k * size.y + j) * size.x + i);
    }

    double& at(std::size_t i, std::size_t j, std::size_t k) {
        return values.at((k * size.y + j) * size.x + i);
    }
};

}  // namespace shardsim
