#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_set>
#include <vector>
#include <cmath>

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

PYBIND11_MODULE(block_cpp, m) {
    m.def("block_children", &block_children_cpp, "C++ version of block_children");
}