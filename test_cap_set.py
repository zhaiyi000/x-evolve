import numpy as np
import itertools
from typing import List, Tuple
import math
from collections import Counter


def evaluate(n: int) -> int:
    """Returns the size of an `n`-dimensional cap set."""
    capset = solve(n)
    return len(capset)


def solve(n: int) -> np.ndarray:
    """Returns a large cap set in `n` dimensions."""
    all_vectors = np.array(list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int32)

    # Powers in decreasing order for compatibility with `itertools.product`, so
    # that the relationship `i = all_vectors[i] @ powers` holds for all `i`.
    powers = np.array([3 ** i for i in range(n - 1, -1, -1)], dtype=np.int32)

    # Precompute all priorities.
    priorities = np.array([priority(tuple(vector)) for vector in all_vectors], dtype=np.float32)

    # Build `capset` greedily, using priorities for prioritization.
    capset = np.empty(shape=(0, n), dtype=np.int32)
    while np.any(priorities != -np.inf):
        # Add a vector with maximum priority to `capset`, and set priorities of 
        # invalidated vectors to `-inf`, so that they never get selected.
        max_index = np.argmax(priorities)
        vector = all_vectors[None, max_index] # [1, n]
        blocking = np.einsum('cn,n->c', (- capset - vector) % 3, powers) # [C]
        priorities[blocking] = -np.inf
        priorities[max_index] = -np.inf
        capset = np.concatenate([capset, vector], axis=0)

    return capset

def tunable(ls):
    return ls[0]

def priority(el: tuple[int, ...]) -> float:
    """Improved version of `priority_v1`."""
    def tunable(options):
        return options[0]  # Return the first element, effectively disabling tuning for now.

    def count_adjacent_pairs(vector):
        """Counts the number of adjacent pairs with the same value."""
        count = 0
        for i in range(len(vector) - 1):
            if vector[i] == vector[i+1]:
                count += 1
        return count

    def count_value(vector, val):
        return vector.count(val)

    def entropy(vector):
        """Calculates the entropy of the vector."""
        counts = [count_value(vector, i) for i in range(3)]
        total = sum(counts)
        probabilities = [count / total for count in counts]
        return -sum(p * math.log(p + 1e-9) for p in probabilities if p > 0)

    def balance_score(vector):
        """Calculates a balance score based on the difference between counts of 0s and 2s."""
        num_zeros = count_value(vector, 0)
        num_twos = count_value(vector, 2)
        return abs(num_zeros - num_twos)

    def centrality(vector):
        """Measures how central the vector is by summing the absolute differences from the center (1)."""
        return sum(abs(x - 1) for x in vector)

    def subvector_sum(vector):
        """Calculates the sum of the subvector."""
        return sum(vector)

    def pairwise_distance(vector, cap_set):
        """Calculates the average pairwise distance between the vector and all vectors in the cap set."""
        if not cap_set:
            return 0
        total_distance = 0
        for vec in cap_set:
            total_distance += sum(abs(a - b) for a, b in zip(vector, vec))
        return total_distance / len(cap_set)

    def pattern_penalty(vector):
        """Penalizes vectors that contain specific patterns that are more likely to form arithmetic lines."""
        penalty = 0
        # Example patterns: 012, 210, 002, 200, 111
        patterns = [(0, 1, 2), (2, 1, 0), (0, 0, 2), (2, 0, 0), (1, 1, 1)]
        for i in range(len(vector) - 2):
            if vector[i:i+3] in patterns:
                penalty += 1
        return penalty

    def weighted_sum(vector):
        """Calculates a weighted sum of the components."""
        weights = tunable([2.0, 1.5, 1.0])
        return sum(w * v for w, v in zip(weights, vector))

    def rotation_symmetry(vector):
        """Calculates the rotational symmetry of the vector."""
        rotations = [vector[i:] + vector[:i] for i in range(len(vector))]
        unique_rotations = len(set(rotations))
        return unique_rotations

    def alternating_pattern_score(vector):
        """Calculates the score based on alternating patterns."""
        score = 0
        for i in range(len(vector) - 1):
            if vector[i] != vector[i + 1]:
                score += 1
        return score

    def imbalance_penalty(vector):
        """Penalizes vectors with high imbalance in 0s, 1s, and 2s."""
        counts = [count_value(vector, i) for i in range(3)]
        max_count = max(counts)
        min_count = min(counts)
        return max_count - min_count

    # Feature 1: Number of '1's. Vectors with more 1s might be more central and therefore interfere more.
    num_ones = count_value(el, 1)
    priority = tunable([1.5, 1.0, 0.5]) * num_ones

    # Feature 2: Balance score based on the difference between counts of 0s and 2s.
    priority += tunable([2.0, 1.5, 1.0]) * balance_score(el)

    # Feature 3: Count adjacent pairs with the same value. Penalize vectors with many adjacent similar values.
    adjacent_pairs = count_adjacent_pairs(el)
    priority += tunable([-1.0, -0.5, 0.0]) * adjacent_pairs

    # Feature 4: Distance from the "center" vector (all 1s). Encourage diversity.
    distance_from_center = centrality(el)
    priority += tunable([-0.3, -0.2, -0.1]) * distance_from_center

    # Feature 5: Sum of subvectors. Averages out the value of a given vector coordinate position.
    priority += tunable([0.2, 0.15, 0.1]) * subvector_sum(el)

    # Feature 6: Number of '2's * number of '0's. Penalizes unbalanced vectors.
    num_zeros = count_value(el, 0)
    num_twos = count_value(el, 2)
    priority += tunable([-0.15, -0.1, -0.05]) * num_zeros * num_twos

    # Feature 7: Heavily penalize vectors with many 0s OR many 2s. This helps prune the search space.
    if num_zeros > tunable([6, 5, 4]):
        priority += tunable([-25, -20, -15])
    if num_twos > tunable([5, 4, 3]):
        priority += tunable([-15, -10, -5])

    # Feature 8: Entropy of the vector. Higher entropy might indicate more randomness, which could be beneficial.
    vector_entropy = entropy(el)
    priority += tunable([0.05, 0.03, 0.01]) * vector_entropy

    # Feature 9: Average pairwise distance to vectors in the current cap set.
    cap_set = []  # Placeholder for the current cap set
    pairwise_dist = pairwise_distance(el, cap_set)
    priority += tunable([0.2, 0.15, 0.1]) * pairwise_dist

    # Feature 10: Pattern penalty. Penalize vectors that contain specific patterns.
    pattern_pen = pattern_penalty(el)
    priority += tunable([-1.5, -1.0, -0.5]) * pattern_pen

    # Feature 11: Weighted sum of the components.
    weighted_sum_val = weighted_sum(el)
    priority += tunable([0.1, 0.05, 0.0]) * weighted_sum_val

    # Feature 12: Rotational symmetry. Vectors with higher symmetry are less likely to form arithmetic lines.
    rotational_symmetry = rotation_symmetry(el)
    priority += tunable([0.05, 0.03, 0.01]) * rotational_symmetry

    # Feature 13: Alternating pattern score. Vectors with alternating patterns are more diverse.
    alternating_score = alternating_pattern_score(el)
    priority += tunable([0.1, 0.05, 0.0]) * alternating_score

    # Feature 14: Imbalance penalty. Vectors with high imbalance are less likely to form arithmetic lines.
    imbalance_pen = imbalance_penalty(el)
    priority += tunable([-0.1, -0.05, 0.0]) * imbalance_pen

    return priority

print(evaluate(7))