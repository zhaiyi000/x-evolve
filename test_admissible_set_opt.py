"""Finds large admissible sets."""

import itertools
import math
import numpy as np
import cpp_helper


def block_children(
    scores: np.ndarray, admissible_set: np.ndarray, new_element: np.ndarray
) -> None:
    """Modifies `scores` to -inf for elements blocked by `new_element`."""
    n = admissible_set.shape[-1]
    powers = 3 ** np.arange(n - 1, -1, -1)

    invalid_vals_raw = {
        (0, 0): (0,),
        (0, 1): (1,),
        (0, 2): (2,),
        (1, 0): (1,),
        (1, 1): (0, 1, 2),
        (1, 2): (1, 2),
        (2, 0): (2,),
        (2, 1): (1, 2),
        (2, 2): (0, 1, 2),
    }
    invalid_vals = [
        [np.array(invalid_vals_raw[(i, j)], dtype=np.int32) for j in range(3)]
        for i in range(3)
    ]

    # Block 2**w elements with the same support as `new_element`.
    w = np.count_nonzero(new_element)
    all_12s = np.array(list(itertools.product((1, 2), repeat=w)), dtype=np.int32)
    blocking = np.einsum("aw,w->a", all_12s, powers[new_element != 0])
    scores[blocking] = -np.inf

    # Block elements disallowed by a pair of an extant point and `new_element`.
    for extant_element in admissible_set:
        blocking = np.zeros(shape=(1,), dtype=np.int32)
        for e1, e2, power in zip(extant_element, new_element, powers):
            blocking = (
                blocking[:, None] + (invalid_vals[e1][e2] * power)[None, :]
            ).ravel()
        scores[blocking] = -np.inf


def solve(n: int, w: int) -> np.ndarray:
    """Generates a constant-weight admissible set I(n, w)."""
    # children = np.array(list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int32)

    # scores = -np.inf * np.ones((3**n,), dtype=np.float32)
    # for child_index, child in enumerate(children):
    #     if sum(child == 0) == n - w:
    #         scores[child_index] = priority(np.array(child))

    # import pickle
    # with open('admissible_set_scores.pkl', 'wb') as f:
    #     pickle.dump((children, scores), f)
    # exit()
    
    import pickle
    with open('admissible_set_scores.pkl', 'rb') as f:
        children, scores = pickle.load(f)
    for child_index, child in enumerate(children):
        if scores[child_index] == 0:
            scores[child_index] = priority(np.array(child))

    max_admissible_set = np.empty((0, n), dtype=np.int32)
    while np.any(scores != -np.inf):
        # Find element with largest score.
        max_index = np.argmax(scores)
        child = children[max_index]
        # block_children(scores, max_admissible_set, child)
        cpp_helper.block_children(scores, max_admissible_set, child)
        max_admissible_set = np.concatenate([max_admissible_set, child[None]], axis=0)

    return max_admissible_set


# @funsearch.run
def evaluate(kargs) -> int:
    """Returns the size of the constructed admissible set."""
    return len(solve(kargs['n'], kargs['w']))


def tunable(ls):
    return ls[0]

# @funsearch.evolve
def priority(el: np.ndarray) -> float:
    """A radically different priority function that considers vector "coverage" and diversity."""

    def calculate_coverage_score(element: np.ndarray, current_set: list[np.ndarray]) -> float:
        """Calculates a score based on how much of the 'solution space' the element covers that isn't already covered."""
        if not current_set:
            return 1.0  # First element gets maximum coverage score

        covered_positions = set()
        for existing_element in current_set:
            for i in range(len(element)):
                if element[i] != 0 or existing_element[i] != 0: # Consider positions covered if either element or existing element uses them
                    covered_positions.add(i)

        newly_covered_positions = set()
        for i in range(len(element)):
            if element[i] != 0:
                is_already_covered = False

                for existing_element in current_set:
                    if existing_element[i] != 0:
                        is_already_covered = True
                        break
                if not is_already_covered:
                    newly_covered_positions.add(i)



        coverage_score = tunable([len(newly_covered_positions) / 12, len(newly_covered_positions)]) # Normalize or not
        return coverage_score

    def calculate_diversity_score(element: np.ndarray, current_set: list[np.ndarray]) -> float:
        """Calculates a score based on how different the element is from existing elements in the set."""
        if not current_set:
            return 1.0  # First element is maximally diverse
        
        similarity_scores = []
        for existing_element in current_set:
            similarity = np.sum(element == existing_element) / len(element)  # Simple similarity: proportion of positions with same value
            similarity_scores.append(similarity)

        average_similarity = np.mean(similarity_scores)
        diversity_score = 1 - average_similarity # Invert to get a diversity score (higher is more diverse)
        return diversity_score



    def calculate_pairwise_distance(element: np.ndarray, current_set: list[np.ndarray]) -> float:

        if not current_set:
            return 1.0
        distance_sum  = 0
        for existing_element in current_set:
            #Hamming Distance as a option for measuring pairwise distance between vectors
            distance_sum += np.sum(element != existing_element)
        average_distance = distance_sum / len(current_set)
        normalized_distance = average_distance / len(element)
        return normalized_distance

    def calculate_balance_score(element: np.ndarray) -> float:
        """Encourages a balance of 1s and 2s within the vector."""
        num_ones = np.sum(element == 1)
        num_twos = np.sum(element == 2)
        total_non_zeros = num_ones + num_twos

        if total_non_zeros == 0:
            return 0

        # Closer to 0.5 means more balanced
        balance = abs(num_ones - num_twos) / total_non_zeros #Ratio of difference, lower is better so invert
        balance_score  = 1- balance
        return balance_score


    # --- Main Priority Calculation ---
    current_set = [] # In a real application this info is passed in, but here we keep track for iterative testing.
    coverage = calculate_coverage_score(el, current_set)
    diversity = calculate_diversity_score(el, current_set)
    balance = calculate_balance_score(el)
    distance = calculate_pairwise_distance(el, current_set)



    # Combine scores with weights: main tunable area!!


    priority = (
        tunable([0.4, .3, .5, .35]) * coverage +
        tunable([0.3, .2, .4, .25])  * diversity +
        tunable([0.1, .3, .2, .15]) * balance+
        tunable([0.2, .1, .1, .25]) * distance
    )
    current_set.append(el)

    return priority

print(evaluate({'n': 12, 'w': 7}))