#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <set>
#include <pybind11/stl.h>
#include <tuple>
#include <bitset>

namespace py = pybind11;

py::array_t<float> block_children_cpp(py::array_t<float> scores,
                                      py::array_t<int> admissible_set,
                                      py::array_t<int> new_element) {
    auto buf_scores = scores.mutable_unchecked<1>();
    auto buf_admissible = admissible_set.unchecked<2>();
    auto buf_new = new_element.unchecked<1>();

    int n = buf_new.shape(0);
    std::vector<int> powers(n);
    for (int i = 0; i < n; ++i)
        powers[i] = std::pow(3, n - 1 - i);

    // Block same support elements
    int w = 0;
    for (int i = 0; i < n; ++i) if (buf_new(i) != 0) w++;

    int num_combinations = std::pow(2, w);
    for (int i = 0; i < num_combinations; ++i) {
        int index = 0;
        int bit_idx = 0;
        for (int j = 0; j < n; ++j) {
            if (buf_new(j) != 0) {
                int val = ((i >> bit_idx) & 1) + 1;
                index += powers[j] * val;
                bit_idx++;
            }
        }
        buf_scores(index) = -INFINITY;
    }

    // Blocking via invalid values
    const std::vector<std::vector<std::vector<int>>> invalid_vals = {
        {{0}, {1}, {2}},
        {{1}, {0, 1, 2}, {1, 2}},
        {{2}, {1, 2}, {0, 1, 2}},
    };

    int num_rows = buf_admissible.shape(0);
    for (int r = 0; r < num_rows; ++r) {
        std::vector<int> block_indices = {0};
        for (int j = 0; j < n; ++j) {
            int e1 = buf_admissible(r, j);
            int e2 = buf_new(j);
            std::vector<int> new_indices;
            for (int base : block_indices) {
                for (int val : invalid_vals[e1][e2]) {
                    new_indices.push_back(base + val * powers[j]);
                }
            }
            block_indices = new_indices;
        }
        for (int idx : block_indices)
            buf_scores(idx) = -INFINITY;
    }

    return scores;
}

// 使用静态局部变量，确保只初始化一次 RNG（性能更好）
int weighted_sample(py::array_t<double> probs) {
    auto buf = probs.request();
    double* ptr = static_cast<double*>(buf.ptr);
    size_t n = buf.shape[0];

    std::vector<double> cdf(n);
    cdf[0] = ptr[0];
    for (size_t i = 1; i < n; ++i)
        cdf[i] = cdf[i - 1] + ptr[i];

    // Normalize CDF to [0, 1]
    double total = cdf[n - 1];
    for (size_t i = 0; i < n; ++i)
        cdf[i] /= total;

    // 使用现代 C++ 随机数生成器
    static thread_local std::mt19937 gen(std::random_device{}());
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);

    double r = dist(gen);
    return std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin();
}


// struct TupleHash {
//     std::size_t operator()(const std::tuple<int, int, int>& t) const {
//         auto [a, b, c] = t;
//         return std::hash<int>()(a) ^ std::hash<int>()(b) << 1 ^ std::hash<int>()(c) << 2;
//     }
// };


static const std::vector<int> INT_TO_WEIGHT = {0, 1, 1, 2, 2, 3, 3};
// static const std::unordered_set<std::tuple<int, int, int>, TupleHash> BAD_TRIPLES = {
//     std::make_tuple(0, 0, 0), std::make_tuple(0, 1, 1), std::make_tuple(0, 2, 2),
//     std::make_tuple(0, 3, 3), std::make_tuple(0, 4, 4), std::make_tuple(0, 5, 5),
//     std::make_tuple(0, 6, 6), std::make_tuple(1, 1, 1), std::make_tuple(1, 1, 2),
//     std::make_tuple(1, 2, 2), std::make_tuple(1, 2, 3), std::make_tuple(1, 2, 4),
//     std::make_tuple(1, 3, 3), std::make_tuple(1, 4, 4), std::make_tuple(1, 5, 5),
//     std::make_tuple(1, 6, 6), std::make_tuple(2, 2, 2), std::make_tuple(2, 3, 3),
//     std::make_tuple(2, 4, 4), std::make_tuple(2, 5, 5), std::make_tuple(2, 6, 6),
//     std::make_tuple(3, 3, 3), std::make_tuple(3, 3, 4), std::make_tuple(3, 4, 4),
//     std::make_tuple(3, 4, 5), std::make_tuple(3, 4, 6), std::make_tuple(3, 5, 5),
//     std::make_tuple(3, 6, 6), std::make_tuple(4, 4, 4), std::make_tuple(4, 5, 5),
//     std::make_tuple(4, 6, 6), std::make_tuple(5, 5, 5), std::make_tuple(5, 5, 6),
//     std::make_tuple(5, 6, 6), std::make_tuple(6, 6, 6)
// };
// static const std::unordered_set<int> BAD_TRIPLES_ENCODED = {
//     0*49 + 0*7 + 0,  // 0
//     0*49 + 1*7 + 1,  // 8
//     0*49 + 2*7 + 2,  // 16
//     0*49 + 3*7 + 3,  // 24
//     0*49 + 4*7 + 4,  // 32
//     0*49 + 5*7 + 5,  // 40
//     0*49 + 6*7 + 6,  // 48
//     1*49 + 1*7 + 1,  // 57
//     1*49 + 1*7 + 2,  // 58
//     1*49 + 2*7 + 2,  // 66
//     1*49 + 2*7 + 3,  // 67
//     1*49 + 2*7 + 4,  // 68
//     1*49 + 3*7 + 3,  // 75
//     1*49 + 4*7 + 4,  // 84
//     1*49 + 5*7 + 5,  // 93
//     1*49 + 6*7 + 6,  // 102
//     2*49 + 2*7 + 2,  // 116
//     2*49 + 3*7 + 3,  // 125
//     2*49 + 4*7 + 4,  // 134
//     2*49 + 5*7 + 5,  // 143
//     2*49 + 6*7 + 6,  // 152
//     3*49 + 3*7 + 3,  // 174
//     3*49 + 3*7 + 4,  // 175
//     3*49 + 4*7 + 4,  // 183
//     3*49 + 4*7 + 5,  // 184
//     3*49 + 4*7 + 6,  // 185
//     3*49 + 5*7 + 5,  // 192
//     3*49 + 6*7 + 6,  // 201
//     4*49 + 4*7 + 4,  // 222
//     4*49 + 5*7 + 5,  // 231
//     4*49 + 6*7 + 6,  // 240
//     5*49 + 5*7 + 5,  // 270
//     5*49 + 5*7 + 6,  // 271
//     5*49 + 6*7 + 6,  // 279
//     6*49 + 6*7 + 6   // 318
// };


// static const std::bitset<343> BAD_TRIPLES_BITMAP = [] {
//     std::bitset<343> b;
//     b.set(0);   // 0*49 + 0*7 + 0
//     b.set(8);   // 0*49 + 1*7 + 1
//     b.set(16);  // ...
//     b.set(24);
//     b.set(32);
//     b.set(40);
//     b.set(48);
//     b.set(57);
//     b.set(58);
//     b.set(66);
//     b.set(67);
//     b.set(68);
//     b.set(75);
//     b.set(84);
//     b.set(93);
//     b.set(102);
//     b.set(116);
//     b.set(125);
//     b.set(134);
//     b.set(143);
//     b.set(152);
//     b.set(174);
//     b.set(175);
//     b.set(183);
//     b.set(184);
//     b.set(185);
//     b.set(192);
//     b.set(201);
//     b.set(222);
//     b.set(231);
//     b.set(240);
//     b.set(270);
//     b.set(271);
//     b.set(279);
//     b.set(318);
//     return b;
// }();
static const std::bitset<343> BAD_TRIPLES_BITMAP = [] {
    std::bitset<343> b;
    std::vector<std::tuple<int, int, int>> triples = {
        {0, 0, 0}, {0, 1, 1}, {0, 2, 2}, {0, 3, 3}, {0, 4, 4}, {0, 5, 5}, {0, 6, 6},
        {1, 1, 1}, {1, 1, 2}, {1, 2, 2}, {1, 2, 3}, {1, 2, 4}, {1, 3, 3},
        {1, 4, 4}, {1, 5, 5}, {1, 6, 6}, {2, 2, 2}, {2, 3, 3}, {2, 4, 4},
        {2, 5, 5}, {2, 6, 6}, {3, 3, 3}, {3, 3, 4}, {3, 4, 4}, {3, 4, 5},
        {3, 4, 6}, {3, 5, 5}, {3, 6, 6}, {4, 4, 4}, {4, 5, 5}, {4, 6, 6},
        {5, 5, 5}, {5, 5, 6}, {5, 6, 6}, {6, 6, 6}
    };

    for (const auto& [a, b_val, c] : triples) {
        int key = a * 49 + b_val * 7 + c;
        b.set(key);
    }
    return b;
}();

std::vector<size_t> get_surviving_children(
    const std::vector<std::vector<int>>& extant_elements,
    const std::vector<int>& new_element,
    const std::vector<std::vector<int>>& valid_children) {

    std::vector<size_t> valid_indices;

    for (size_t idx = 0; idx < valid_children.size(); ++idx) {
        const auto& child = valid_children[idx];

        // Check first condition
        bool all_le = true;
        for (size_t i = 0; i < new_element.size(); ++i) {
            if (INT_TO_WEIGHT[new_element[i]] > INT_TO_WEIGHT[child[i]]) {
                all_le = false;
                break;
            }
        }
        if (all_le) continue;

        // Check second condition
        bool all_ge = true;
        for (size_t i = 0; i < new_element.size(); ++i) {
            if (INT_TO_WEIGHT[new_element[i]] < INT_TO_WEIGHT[child[i]]) {
                all_ge = false;
                break;
            }
        }
        if (all_ge) continue;

        // Check third condition
        bool invalid = false;
        for (const auto& extant : extant_elements) {
            bool all_bad = true;
            for (size_t i = 0; i < extant.size(); ++i) {
                // std::array<int, 3> triple = {extant[i], new_element[i], child[i]};
                // std::sort(triple.begin(), triple.end());
                // auto tpl = std::make_tuple(triple[0], triple[1], triple[2]);
                int a = extant[i], b = new_element[i], c = child[i];
                if (a > b) std::swap(a, b);
                if (b > c) std::swap(b, c);
                if (a > b) std::swap(a, b);
                // auto tpl = std::make_tuple(a, b, c);
                int key = a * 49 + b * 7 + c;
                // if (BAD_TRIPLES_ENCODED.find(key) == BAD_TRIPLES_ENCODED.end()) {
                if (!BAD_TRIPLES_BITMAP[key]) {
                    all_bad = false;
                    break;
                }
            }
            if (all_bad) {
                invalid = true;
                break;
            }
        }
        if (invalid) continue;

        valid_indices.push_back(idx);
    }

    return valid_indices;
}

std::vector<std::vector<int>> greedy_search(
    int num_groups,
    const std::vector<float>& valid_scores,
    const std::vector<std::vector<int>>& valid_children) {

    std::vector<std::vector<int>> pre_admissible;
    std::vector<std::vector<int>> current_children = valid_children;
    std::vector<float> current_scores = valid_scores;

    while (!current_children.empty()) {
        auto max_it = std::max_element(current_scores.begin(), current_scores.end());
        size_t max_idx = std::distance(current_scores.begin(), max_it);
        std::vector<int> max_child = current_children[max_idx];

        auto survivors = get_surviving_children(pre_admissible, max_child, current_children);

        std::vector<std::vector<int>> new_children;
        std::vector<float> new_scores;
        for (auto i : survivors) {
            new_children.push_back(current_children[i]);
            new_scores.push_back(current_scores[i]);
        }

        current_children = new_children;
        current_scores = new_scores;

        pre_admissible.push_back(max_child);
    }

    return pre_admissible;
}


PYBIND11_MODULE(cpp_helper, m) {
    m.def("sample", &weighted_sample, "Weighted random sampler using C++11 RNG");
    m.def("block_children", &block_children_cpp, "C++ version of block_children");
    
    m.def("greedy_search", [](int num_groups, py::array_t<float> scores, const std::vector<py::array_t<int>>& children) {
        // Convert scores
        auto scores_buf = scores.unchecked<1>();
        std::vector<float> scores_vec(scores_buf.data(0), scores_buf.data(0) + scores_buf.size());

        // Convert children
        std::vector<std::vector<int>> children_vec;
        for (const auto& child : children) {
            auto child_buf = child.unchecked<1>();
            children_vec.emplace_back(child_buf.data(0), child_buf.data(0) + child_buf.size());
        }

        auto result = greedy_search(num_groups, scores_vec, children_vec);

        // Convert to numpy array
        py::array_t<int> output({(int)result.size(), (int)result[0].size()});
        auto output_buf = output.mutable_unchecked<2>();
        for (size_t i = 0; i < result.size(); ++i) {
            for (size_t j = 0; j < result[i].size(); ++j) {
                output_buf(i, j) = result[i][j];
            }
        }
        return output;
    }, py::arg("num_groups"), py::arg("valid_scores"), py::arg("valid_children"));
}