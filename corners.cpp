#include <vector>
#include <array>
// #include <unordered_set>
#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// 定义维度和模数
int n = 4; // 维度
int p = 2; // 模数

using Vector = std::vector<int>;
namespace py = pybind11;
// 哈希函数用于存储向量
// struct VectorHash {
//     std::size_t operator()(const Vector& v) const {
//         std::size_t seed = 0;
//         for (int val : v) {
//             seed ^= std::hash<int>{}(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
//         }
//         return seed;
//     }
// };
void set_params(int n_val, int p_val) {
    n = n_val;
    p = p_val;
}

std::vector<Vector> generate_all_vectors() {
    std::vector<Vector> result;
    Vector vec(2 * n, 0);

    while (true) {
        result.push_back(vec);

        // 类似于 base-p 进位操作（从最后一位开始累加）
        int i = 2 * n - 1;
        while (i >= 0) {
            vec[i]++;
            if (vec[i] < p) break;
            vec[i] = 0;
            i--;
        }

        if (i < 0) break;  // 所有组合已生成完毕
    }

    return result;
}

// 检查三个向量是否形成角
bool forms_corner(const Vector& a, const Vector& b, const Vector& c) {
    Vector lambda(n, 0);
    bool match_b = true, match_c = true;
    for (int i = 0; i < n; ++i) {
        // bool match_a_2 = true, match_c_2 = true;
        // bool match_a_3 = true, match_b_3 = true;
        int j = 0;
        for (j = 0; j < p; ++j) {
            if ((a[i] + j) % p != b[i] || a[i + n] != b[i + n]) {
                match_b = false;
                continue;
            }
            if (a[i] != c[i] || (a[i + n] + j) % p != c[i + n]) {
                match_c = false;
                continue;
            }
            lambda[i] = j;
            match_b = true;
            match_c = true;
            break;
        }
        if (!match_b || !match_c) return false;
    }
    for (int i = 0; i < n; ++i) {
        if (lambda[i] != 0) return true;
    }
    return false;
}

// 检查新向量是否可以加入当前集合而不形成角
bool is_valid(const Vector& candidate, const std::vector<Vector>& current_set) {
    for (const auto& v1 : current_set) {
        for (const auto& v2 : current_set) {
            if (v1 != v2) {
                if (forms_corner(candidate, v1, v2)) return false;
                if (forms_corner(v1, candidate, v2)) return false;
                if (forms_corner(v1, v2, candidate)) return false;
            }
        }
    }
    return true;
}

// 贪心算法构建角自由集
std::vector<Vector> greedy(const std::vector<float>& scores) {
    std::vector<Vector> selected;
    std::vector<Vector> all_vectors = generate_all_vectors();
    std::vector<size_t> indices(all_vectors.size());
    std::iota(indices.begin(), indices.end(), 0);
    // 根据分数降序排序
    std::sort(indices.begin(), indices.end(), [&scores](size_t i1, size_t i2) {
        return scores[i1] > scores[i2];
    });
    for (int i = 0; i < 30; ++i) {
        std::cout << indices[i] << " ";
    }
    for (size_t idx : indices) {
        const Vector& candidate = all_vectors[idx];
        if (is_valid(candidate, selected)) {
            std::cout << idx << " ";
            for (int x : candidate) std::cout << x << " ";
            std::cout << "\n";
            selected.push_back(candidate);
        }
    }

    return selected;
}

PYBIND11_MODULE(corners, m) {
    m.def("set_params", &set_params, "Set parameters n and p");
    m.def("greedy", &greedy, "Greedy algorithm for corner-free set selection");
}
