import itertools
import numpy as np
import pickle
import time
# @funsearch.run
def evaluate(params: dict) -> int:
    """Returns the size of an independent set."""
    independent_set = solve(params['num_nodes'], params['n'])
    return len(independent_set)


def solve(num_nodes: int, n: int) -> list[tuple[int, ...]]:
    """Gets independent set with maximal size.

    Args:
        num_nodes: The number of nodes of the base cyclic graph.
        n: The power we raise the graph to.

    Returns:
        A list of `n`-tuples in `{0, 1, 2, ..., num_nodes - 1}`.
    """
    to_block = np.array(list(itertools.product([-1, 0, 1], repeat=n)))

    # Powers in decreasing order for compatibility with `itertools.product`, so
    # that the relationship `i = children[i] @ powers` holds for all `i`.
    powers = num_nodes ** np.arange(n - 1, -1, -1)

    # Precompute the priority scores.
    children = np.array(
        list(itertools.product(range(num_nodes), repeat=n)), dtype=np.int32)
    with open('cycle_graphs15_4.pkl', 'wb') as f:
        pickle.dump((to_block, powers, children), f)
    return [0,0]
    with open('cycle_graphs15_4.pkl', 'rb') as f:
        to_block, powers, children = pickle.load(f)
    print(to_block.shape)
    scores = np.array([priority(tuple(child), num_nodes, n)
                        for child in children],dtype=np.float32)

    # Build `max_set` greedily, using scores for prioritization.
    max_set = np.empty(shape=(0, n), dtype=np.int32)
    while np.any(scores != -np.inf):
        # Add a child with a maximum score to `max_set`, and set scores of
        # invalidated children to -inf, so that they never get selected.
        max_index = np.argmax(scores)
        child = children[None, max_index]  # [1, n]

        blocking = np.einsum(
            'cn,n->c', (to_block + child) % num_nodes, powers)  # [C]
        scores[blocking] = -np.inf
        max_set = np.concatenate([max_set, child], axis=0)

    return [tuple(map(int, el)) for el in max_set]


# @funsearch.evolve
def priority(el: tuple[int, ...], num_nodes: int, n: int) -> float:
    """Returns the priority with which we want to add `el` to the set.

    Args:
        el: an n-tuple representing the element to consider whether to add.
        num_nodes: the number of nodes of the base graph.
        n: an integer, power of the graph.

    Returns:
        A number reflecting the priority with which we want to add `el` to the
        independent set.
    """
    score = 0.
    for i in range(n):
        if el[i] == el[(i + 2) % n]:
            score += 1
        else:
            score -= 1
        x = ((n - 2) * el[i] - el[(i + 1) % n] - el[(i + 2) % n] - (n + 1) * el[(i + 3) % n]) % num_nodes
        score -= 0.5 * (x - el[(i + 1) % n]) ** 2
        score += 0.1 * (num_nodes - 1 - (x - 1) % num_nodes) ** 2
        score += 0.2 * (num_nodes - 1 - (x - 2) % num_nodes) ** 2
    return score

begin = time.time()
print(evaluate({'num_nodes': 15, 'n': 4}))
end = time.time()
print(end - begin)