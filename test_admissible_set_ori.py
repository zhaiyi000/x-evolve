"""Finds large admissible sets."""

import itertools
import math
import numpy as np


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
    children = np.array(list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int32)

    scores = -np.inf * np.ones((3**n,), dtype=np.float32)
    for child_index, child in enumerate(children):
        if sum(child == 0) == n - w:
            scores[child_index] = priority(np.array(child), n, w)

    max_admissible_set = np.empty((0, n), dtype=np.int32)
    while np.any(scores != -np.inf):
        # Find element with largest score.
        max_index = np.argmax(scores)
        child = children[max_index]
        block_children(scores, max_admissible_set, child)
        max_admissible_set = np.concatenate([max_admissible_set, child[None]], axis=0)

    return max_admissible_set


# @funsearch.run
def evaluate(n: int, w: int) -> int:
    """Returns the size of the constructed admissible set."""
    return len(solve(n, w))


# @funsearch.evolve
def priority(el: tuple[int, ...], n: int, w: int) -> float:
    """Returns the priority with which we want to add `el` to the set."""
    return 0.0


print(evaluate(n=12, w=7))


def priority(el: tuple[int, ...], n: int, w: int) -> float:
    score = 0.0
    for i in range(n):
        if el[i] == 1:
            score -= 0.9 ** (i % 4)
        if el[i] == 2:
            score -= 0.98 ** (30 - (i % 4))
        if el[i] == 1 and el[i - 4] == 1:
            score -= 0.98 ** (30 - (i % 4))
        if el[i] == 2 and el[i - 4] != 0:
            score -= 0.98 ** (30 - (i % 4))
        if el[i] == 2 and el[i - 4] == 1 and el[i - 8] == 2:
            score -= 0.98 ** (30 - (i % 4))
            score -= 6.3
        if el[i] == 2 and el[i - 4] == 2 and el[i - 8] == 1:
            score -= 0.98 ** (30 - (i % 4))
        if el[i] == 2 and el[i - 4] == 1 and el[i - 8] == 1:
            score -= 6.3
        if el[i] == 2 and el[i - 4] == 0 and el[i - 8] == 2:
            score -= 6.3
        if el[i] == 1 and el[i - 4] == 1 and el[i - 8] == 0:
            score -= 2.2
    return score


admissible_12_7 = solve(12, 7)
assert admissible_12_7.shape == (math.comb(12, 7), 12)


def compute_capacity_bound(n: int, w: int, size: int, m: int) -> float:
    """Returns the lower bound on the cap set capacity.

    We use discovered admissible sets A(n, w) to construct large cap sets,
    following a recipe analogous to [Edel, 2004] and [Tyrrell, 2022]:
    1. Start with the extendable collection E1 = (A0, A1, A2) of three
       6-dimensional cap sets of respective sizes (a0, a1, a2) = (12, 112, 112).
    2. Apply a recursively admissible set I(m, m - 1) to E1, which results in a
       new extendable collection E2 = (B0, B1, B2) of three 6*m-dimensional cap
       sets of sizes (b0, b1, b2) = (a0 * m * a1 ** (m - 1), a1 ** m, a1 ** m).
    3. Apply the admissible set A(n, w) of size `size` to E2, which results in a
       6*m*n-dimensional cap set C of size `size * (b0 ** (n - w)) * (b1 ** w)`.

    Args:
      n: Dimensionality of the discovered admissible set A(n, w).
      w: The weight of the vectors in the discovered admissible set A(n, w).
      size: The size |A(n, w)| of the discovered admissible set.
      m: Dimensionality of the recursively admissible set I(m, m - 1) to use.
    """
    a0, a1, _ = (12, 112, 112)
    b0 = m * a0 * (a1 ** (m - 1))
    b1 = a1**m
    log_cap_set_size = np.log(size) + (n - w) * np.log(b0) + w * np.log(b1)
    log_capacity = log_cap_set_size / (6 * m * n)
    return np.exp(log_capacity)


compute_capacity_bound(12, 7, len(admissible_12_7), m=7)


def get_cyclic_permutations(partition: list[list[int]]) -> set[tuple[int, ...]]:
    """Returns all combinations of cyclic permutations within `partition`."""
    identity_permutation = list(range(sum(map(len, partition))))
    permutations = set()
    for cyclic_shifts in itertools.product(*[range(len(g)) for g in partition]):
        permutation = list(identity_permutation)
        for group, cyclic_shift in zip(partition, cyclic_shifts):
            for i, x in enumerate(group):
                permutation[x] = group[(i + cyclic_shift) % len(group)]
        permutations.add(tuple(permutation))
    return permutations


# Obtain all independent cyclic permutations of coordinates within each of the
# following four triples of coordinates. There are 3**4=81 such permutations.
coordinate_triples = [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]
permutations = get_cyclic_permutations(coordinate_triples)
assert len(permutations) == 3**4

# Check that permuting coordinates in any of these 81 ways preserves the
# admissible set as a set of vectors, i.e. up to the order of its rows.
original_set = set(map(tuple, admissible_12_7))
for permutation in permutations:
    permuted_set = set(map(tuple, admissible_12_7[:, permutation]))
    assert original_set == permuted_set


"""Finds large symmetric admissible sets."""
import itertools
import math
import numpy as np

TRIPLES = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 2), (0, 2, 1), (1, 1, 1), (2, 2, 2)]
INT_TO_WEIGHT = [0, 1, 1, 2, 2, 3, 3]


def expand_admissible_set(
    pre_admissible_set: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    """Expands a pre-admissible set into an admissible set."""
    num_groups = len(pre_admissible_set[0])
    admissible_set = []
    for row in pre_admissible_set:
        rotations = [[] for _ in range(num_groups)]
        for i in range(num_groups):
            x, y, z = TRIPLES[row[i]]
            rotations[i].append((x, y, z))
            if not x == y == z:
                rotations[i].append((z, x, y))
                rotations[i].append((y, z, x))
        product = list(itertools.product(*rotations))
        concatenated = [sum(xs, ()) for xs in product]
        admissible_set.extend(concatenated)
    return admissible_set


def get_surviving_children(extant_elements, new_element, valid_children):
    """Returns the indices of `valid_children` that remain valid after adding `new_element` to `extant_elements`."""
    bad_triples = set(
        [
            (0, 0, 0),
            (0, 1, 1),
            (0, 2, 2),
            (0, 3, 3),
            (0, 4, 4),
            (0, 5, 5),
            (0, 6, 6),
            (1, 1, 1),
            (1, 1, 2),
            (1, 2, 2),
            (1, 2, 3),
            (1, 2, 4),
            (1, 3, 3),
            (1, 4, 4),
            (1, 5, 5),
            (1, 6, 6),
            (2, 2, 2),
            (2, 3, 3),
            (2, 4, 4),
            (2, 5, 5),
            (2, 6, 6),
            (3, 3, 3),
            (3, 3, 4),
            (3, 4, 4),
            (3, 4, 5),
            (3, 4, 6),
            (3, 5, 5),
            (3, 6, 6),
            (4, 4, 4),
            (4, 5, 5),
            (4, 6, 6),
            (5, 5, 5),
            (5, 5, 6),
            (5, 6, 6),
            (6, 6, 6),
        ]
    )

    # Compute.
    valid_indices = []
    for index, child in enumerate(valid_children):
        # Invalidate based on 2 elements from `new_element` and 1 element from a
        # potential child.
        if all(
            INT_TO_WEIGHT[x] <= INT_TO_WEIGHT[y] for x, y in zip(new_element, child)
        ):
            continue
        # Invalidate based on 1 element from `new_element` and 2 elements from a
        # potential child.
        if all(
            INT_TO_WEIGHT[x] >= INT_TO_WEIGHT[y] for x, y in zip(new_element, child)
        ):
            continue
        # Invalidate based on 1 element from `extant_elements`, 1 element from
        # `new_element`, and 1 element from a potential child.
        is_invalid = False
        for extant_element in extant_elements:
            if all(
                tuple(sorted((x, y, z))) in bad_triples
                for x, y, z in zip(extant_element, new_element, child)
            ):
                is_invalid = True
                break
        if is_invalid:
            continue

        valid_indices.append(index)
    return valid_indices


def solve(n: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Generates a symmetric constant-weight admissible set I(n, w)."""
    num_groups = n // 3
    assert 3 * num_groups == n

    # Compute the scores of all valid (weight w) children.
    valid_children = []
    for child in itertools.product(range(7), repeat=num_groups):
        weight = sum(INT_TO_WEIGHT[x] for x in child)
        if weight == w:
            valid_children.append(np.array(child, dtype=np.int32))
    valid_scores = np.array(
        [priority(sum([TRIPLES[x] for x in xs], ()), n, w) for xs in valid_children]
    )

    # Greedy search guided by the scores.
    pre_admissible_set = np.empty((0, num_groups), dtype=np.int32)
    while valid_children:
        max_index = np.argmax(valid_scores)
        max_child = valid_children[max_index]
        surviving_indices = get_surviving_children(
            pre_admissible_set, max_child, valid_children
        )
        valid_children = [valid_children[i] for i in surviving_indices]
        valid_scores = valid_scores[surviving_indices]

        pre_admissible_set = np.concatenate(
            [pre_admissible_set, max_child[None]], axis=0
        )

    return pre_admissible_set, np.array(expand_admissible_set(pre_admissible_set))


# @funsearch.run
def evaluate(n: int, w: int) -> int:
    """Returns the size of the expanded admissible set."""
    _, admissible_set = solve(n, w)
    return len(admissible_set)


# @funsearch.evolve
def priority(el: tuple[int, ...], n: int, w: int) -> float:
    """Returns the priority with which we want to add `el` to the set."""
    return 0.0


def priority(el: tuple[int, ...], n: int, w: int) -> float:
    score = 0.0
    for i in range(n):
        if el[i] < el[i - 1]:
            score += 1
        elif el[i] < el[i - 2]:
            score += 0.05
        elif el[i] < el[i - 3]:
            score -= 0.05
        elif el[i] < el[i - 4]:
            score += 0.01
        elif el[i] < el[i - 5]:
            score -= 0.01
        elif el[i] < el[i - 6]:
            score += 0.001
        else:
            score += 0.005

    for i in range(n):
        if el[i] == el[i - 1]:
            score -= w
        elif el[i] == 0 and i != n - 1 and el[i + 1] != 0:
            score += w
        if el[i] != el[i - 1]:
            score += w

    for i in range(n):
        if el[i] < el[i - 1]:
            if el[i] == 0:
                score -= w
    return score


pre_admissible_15_10, admissible_15_10 = solve(15, 10)
assert admissible_15_10.shape == (math.comb(15, 10), 15)
assert pre_admissible_15_10.shape == (101, 5)

# Show the resulting lower bound on the cap set capacity.
print(compute_capacity_bound(15, 10, len(admissible_15_10), m=5))


def priority(el: tuple[int, ...], n: int, w: int) -> float:
    score = 0
    coeff = 0
    for pos, x in zip(range(n), el):
        y = (el[(pos + 1) % n] - el[(pos)]) % n
        z = (el[(pos + 2) % n] - el[(pos)]) % n
        p = (el[(pos - 1) % n] + 1) % n

        u = (el[(pos - 2) % n] + 1) % n
        v = (el[(pos + 3) % n] + 1) % n

        score += 3 * p * (p + coeff) * (p + w) + (p + coeff) ** 2 * (w + 1)
        score += 2 * p * v * (p + w) + v * z * (-1 + w) - (p + coeff) * (-1 + w)
        score += (
            v * (u + w)
            + u
            + 3 * u * y * (1 + w)
            + u * z * (w - 1)
            - (p + coeff) * (w - 1)
        )
        score += (1 + w) ** 6 * 3 * coeff**2

    return score


# Uncomment to execute; note it can take ~15 minutes to run this.
# pre_admissible_21_15, admissible_21_15 = solve(21, 15)
# assert admissible_21_15.shape == (43_596, 21)
# assert pre_admissible_21_15.shape == (308, 7)


def priority(el: tuple[int, ...], n: int, w: int) -> float:
    result = 0.0
    for i in range(n):
        n_violations = 0

        if el[i] < el[i - 1]:
            result += (el[i - 1] ** 0.5) * w**2 / (6 * 6)
            n_violations += 1

        if el[i] < el[i - 2]:
            result += el[i - 2] ** 0.5
            n_violations += 1

        if el[i - 1] != 0:
            result -= (el[i] - el[i - 1]) * w**2 / (6 * 3)
            n_violations += 2

        if el[i - 2] != 0:
            result -= (el[i] - el[i - 2]) * w**2 / (6 * 6) * (0.95**n_violations)
            n_violations += 1

        result -= (0.02 ** el[i]) * (el[i] - el[i - 8])

    return result


# Executing this would take ~7 hours.
# pre_admissible_24_17, admissible_24_17 = solve(24, 17)
# assert admissible_24_17.shape == (237_984, 24)
# assert pre_admissible_24_17.shape == (736, 8)
