import corners
import itertools
import numpy as np
import time
# @funsearch.run
def evaluate(n: int)-> int:
    """Returns the size of the maximum set of indices found by DFS."""
    return len(solve(n))
def solve(n: int)-> list[tuple[int, ...]]:
    """Runs DFS to find a large set of indices."""
    # Obtain the priority scores.
    scores = np.array([priority(el, n) for el in itertools.product(range(2), repeat=2 * n)])
    # print(scores)
    all_indices = np.arange(len(scores), dtype=np.int32)
    # Run a greedy approach that iteratively adds the next highest-priority
    # index that guarantees the combinatorial degeneration property.
    corners.set_params(n, 2)
    return corners.greedy(scores)
def tunable(options):
    # implement a simple random choice for demonstration purposes
    # in a real-world scenario, you would use a more sophisticated method
    # to select the optimal value from the options
    import random
    return random.choice(options)
# @funsearch.evolve
def priority(el: tuple[int, ...], n: int)-> float:
    """Returns the priority with which we want to add `el`.
    Args:
        el: A candidate element to be considered, as a tuple of length `2 * n` with
        elements in {0, 1}.
        n: Power of the graph.
    Returns:
        A number reflecting the priority with which we want to add `el` to the set.
    """
    x = el[:n]
    y = el[n:]
    sum_x = sum(x)
    sum_y = sum(y)
    diff_x = n - sum_x
    diff_y = n - sum_y
    xor_sum = sum(a ^ b for a, b in zip(x, y))
    dot_product = sum(a * b for a, b in zip(x, y))
    combined_sum = sum_x + sum_y

    priority = 1.3 * (diff_x + diff_y)
    priority += 0.5 * xor_sum
    priority += -0.1 * dot_product
    priority += 0.7 * combined_sum
    priority += -4.0 * (sum_x == 1)
    priority += 5.0 * (sum_y == n // 2)
    priority += -0.4 * abs(sum_x - sum_y)
    priority += 0.4 * (sum_x > sum_y)

    if sum_x == 0:
        priority += -5
    if sum_y == n // 1:
        priority += 6
    priority -= 0.2 * (sum_x * sum_y)
    priority += -0.05 * (sum_x + sum_y) ** 2

    parity_x = sum_x % 2
    parity_y = sum_y % 2
    if parity_x == 0 and parity_y == 0:
        priority += 4
    else:
        priority -= 0.8
    return priority
    # return 0.
begin = time.time()
print(evaluate(4))
end = time.time()
print(end-begin)