"""Finds large symmetric admissible sets."""

import itertools
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
    bad_triples = set([
      (0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3), (0, 4, 4), (0, 5, 5),
      (0, 6, 6), (1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 2, 3), (1, 2, 4),
      (1, 3, 3), (1, 4, 4), (1, 5, 5), (1, 6, 6), (2, 2, 2), (2, 3, 3),
      (2, 4, 4), (2, 5, 5), (2, 6, 6), (3, 3, 3), (3, 3, 4), (3, 4, 4),
      (3, 4, 5), (3, 4, 6), (3, 5, 5), (3, 6, 6), (4, 4, 4), (4, 5, 5),
      (4, 6, 6), (5, 5, 5), (5, 5, 6), (5, 6, 6), (6, 6, 6)])

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


def greedy_search(
    num_groups: int, valid_scores: np.ndarray, valid_children: list[np.ndarray]
):
    # Greedy search guided by the scores.
    pre_admissible_set = np.empty((0, num_groups), dtype=np.int32)
    while valid_children:
        max_index = np.argmax(valid_scores)
        max_child = valid_children[max_index]
        # surviving_indices = get_surviving_children(pre_admissible_set, max_child,
        #                                            valid_children)
        surviving_indices = get_surviving_children(
            pre_admissible_set, max_child, valid_children
        )
        valid_children = [valid_children[i] for i in surviving_indices]
        valid_scores = valid_scores[surviving_indices]

        pre_admissible_set = np.concatenate(
            [pre_admissible_set, max_child[None]], axis=0
        )

    return pre_admissible_set


def solve(n: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Generates a symmetric constant-weight admissible set I(n, w)."""
    num_groups = n // 3
    assert 3 * num_groups == n
    import cpp_helper

    # # Compute the scores of all valid (weight w) children.
    # valid_children = []
    # for child in itertools.product(range(7), repeat=num_groups):
    #     weight = sum(INT_TO_WEIGHT[x] for x in child)
    #     if weight == w:
    #         valid_children.append(np.array(child, dtype=np.int32))
    
    # valid_children_expand = [sum([TRIPLES[x] for x in xs], ()) for xs in valid_children]    

    # np.save('admissible_set_27_19.npy', np.array(valid_children))
    # np.save('admissible_set_27_19_expand.npy', np.array(valid_children_expand))
    # exit()
    
    valid_children = np.load('admissible_set_27_19.npy')
    valid_children_expand = np.load('admissible_set_27_19_expand.npy')
    valid_children_expand = [tuple(xs) for xs in valid_children_expand.tolist()]
    
    valid_scores = np.array(
        [priority(xs) for xs in valid_children_expand]
    )

    pre_admissible_set = cpp_helper.greedy_search(
        num_groups, valid_scores, valid_children
    )
    return pre_admissible_set, np.array(expand_admissible_set(pre_admissible_set))


# @funsearch.run
def evaluate(kargs) -> int:
    """Returns the size of the expanded admissible set."""
    _, admissible_set = solve(kargs['n'], kargs['w'])
    return len(admissible_set)

def tunable(ls):
    return ls[1]

# # @funsearch.evolve
# def priority(el: tuple[int, ...]) -> float:
#     n = 21
#     w = 15
#     total = sum(el)
#     zeros = el.count(0)
#     ones = el.count(1)
#     twos = el.count(2)
#     score = tunable([total, w - total, zeros, ones, twos])
#     score += tunable([sum(el[:10]), sum(el[10:])])
#     score += tunable([np.mean(el), np.median(el)])
#     score += tunable([np.max(el), np.min(el)])
#     return score

# print(evaluate(dict(n=21, w=15)))
# exit()


def priority(el: tuple[int, ...]) -> float:
    n = 24
    w = 17
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


print(evaluate(dict(n=27, w=19)))
