#include "index/flat/flat_index.h"
#include "core/distance.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>

namespace {

constexpr char kMagic[4] = {'V', 'D', 'B', 'I'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kTypeFlat = 1;

template <typename T>
bool write_pod(std::ofstream& out, const T& v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(T));
    return static_cast<bool>(out);
}

template <typename T>
bool read_pod(std::ifstream& in, T& v) {
    in.read(reinterpret_cast<char*>(&v), sizeof(T));
    return static_cast<bool>(in);
}

} // namespace

FlatIndex::FlatIndex(size_t dim) : dim_(dim) {}

void FlatIndex::add(const Vector& vec) {
    vectors_.push_back(vec);
}

std::vector<size_t> FlatIndex::search(const Vector& query, size_t k) const {
    std::vector<std::pair<float, size_t>> scores;
    scores.reserve(vectors_.size());

    for (size_t i = 0; i < vectors_.size(); i++) {
        float dist = l2_distance(query, vectors_[i]);
        scores.emplace_back(dist, i);
    }

    k = std::min(k, scores.size());
    if (k == 0) {
        return {};
    }

    std::partial_sort(
        scores.begin(),
        scores.begin() + k,
        scores.end()
    );

    std::vector<size_t> result;
    for (size_t i = 0; i < k; i++) {
        result.push_back(scores[i].second);
    }
    return result;
}

size_t FlatIndex::size() const {
    return vectors_.size();
}

bool FlatIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }

    const uint64_t dim_u64 = static_cast<uint64_t>(dim_);
    const uint64_t count_u64 = static_cast<uint64_t>(vectors_.size());

    out.write(kMagic, sizeof(kMagic));
    if (!out ||
        !write_pod(out, kVersion) ||
        !write_pod(out, kTypeFlat) ||
        !write_pod(out, dim_u64) ||
        !write_pod(out, count_u64)) {
        return false;
    }

    for (const auto& vec : vectors_) {
        if (vec.size() != dim_) {
            return false;
        }
        out.write(reinterpret_cast<const char*>(vec.data()), static_cast<std::streamsize>(dim_ * sizeof(float)));
        if (!out) {
            return false;
        }
    }

    return true;
}

bool FlatIndex::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    char magic[4] = {};
    uint32_t version = 0;
    uint32_t type = 0;
    uint64_t dim_u64 = 0;
    uint64_t count_u64 = 0;

    in.read(magic, sizeof(magic));
    if (!in ||
        std::memcmp(magic, kMagic, sizeof(kMagic)) != 0 ||
        !read_pod(in, version) ||
        !read_pod(in, type) ||
        !read_pod(in, dim_u64) ||
        !read_pod(in, count_u64)) {
        return false;
    }

    if (version != kVersion || type != kTypeFlat) {
        return false;
    }

    if (dim_u64 != static_cast<uint64_t>(dim_)) {
        return false;
    }

    const size_t count = static_cast<size_t>(count_u64);
    std::vector<Vector> loaded;
    loaded.resize(count, Vector(dim_));

    for (size_t i = 0; i < count; ++i) {
        in.read(reinterpret_cast<char*>(loaded[i].data()), static_cast<std::streamsize>(dim_ * sizeof(float)));
        if (!in) {
            return false;
        }
    }

    vectors_.swap(loaded);
    return true;
}
