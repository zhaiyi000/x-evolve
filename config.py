import os


config_type = 'cycle_graphs'
log_dir = os.environ.get('LOG_DIR', 'logs')
sample_llm_cnt = 10



if config_type not in ['bin_packing', 'cap_set', 'cycle_graphs']:
    raise Exception('wrong type')


if config_type == 'bin_packing':
    evaluate_function_c_v1 = 0.1
    evaluate_function_c_l1 = 0.1
    evaluate_function_c_1 = 100
    evaluate_function_temperature = 0.1
    evaluate_function_mask_half = False

    sample_iterator_temperature = 1
    sample_iterator_no_update_cnt = 3

    sample_llm_api_min_score = -500

elif config_type == 'cap_set':
    evaluate_function_c_v1 = 0.
    evaluate_function_c_l1 = 0.
    evaluate_function_c_1 = 1
    evaluate_function_temperature = 10
    evaluate_function_mask_half = True

    sample_iterator_temperature = 100
    sample_iterator_no_update_cnt = 5

    sample_llm_api_min_score = 256

elif config_type == 'cycle_graphs':
    evaluate_function_c_v1 = 0.1
    evaluate_function_c_l1 = 0.1
    evaluate_function_c_1 = 10
    evaluate_function_temperature = 1
    evaluate_function_mask_half = True

    sample_iterator_temperature = 100
    sample_iterator_no_update_cnt = 5

    sample_llm_api_min_score = 200

else:
    raise Exception('wrong type')



if config_type == 'bin_packing':
    
    additional_prompt = \
"""
Create an improved Python function for online bin-packing that demonstrates:
Novel priority strategy: Propose a smarter item-bin matching approach considering both spatial fit and future packing potential.
Parameter tuning points: Clearly mark tuning parameters using tunable([option1, option2, ...]) wrapper. Examples:
`if remaining_capacity > tunable([0.2, 0.5]):`
`sorted(items, key=lambda x: tunable([x.size, x.weight]))`
Focus first on strategic innovation, then expose tuning parameters through tunable([option1, option2, ...]) calls. Keep implementation practical but non-trivial.
"""

    specification = r'''
import numpy as np


def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]


def online_binpack(
        items: tuple[float, ...], bins: np.ndarray
) -> tuple[list[list[float, ...], ...], np.ndarray]:
    """Performs online binpacking of `items` into `bins`."""
    # Track which items are added to each bin.
    packing = [[] for _ in bins]
    # Add items to bins.
    for item in items:
        # Extract bins that have sufficient space to fit item.
        valid_bin_indices = get_valid_bin_indices(item, bins)
        # Score each bin based on heuristic.
        priorities = priority(item, bins[valid_bin_indices])
        # Add item to bin with highest priority.
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)
    # Remove unused bins from packing.
    packing = [bin_items for bin_items in packing if bin_items]
    return packing, bins


@funsearch.run
def evaluate(instances: dict) -> float:
    """Evaluate heuristic function on a set of online binpacking instances."""
    # List storing number of bins used for each instance.
    num_bins = []
    # Perform online binpacking for each instance.
    for name in instances:
        instance = instances[name]
        capacity = instance['capacity']
        items = instance['items']
        # Create num_items bins so there will always be space for all items,
        # regardless of packing order. Array has shape (num_items,).
        bins = np.array([capacity for _ in range(instance['num_items'])])
        # Pack items into bins and return remaining capacity in bins_packed, which
        # has shape (num_items,).
        _, bins_packed = online_binpack(items, bins)
        # If remaining capacity in a bin is equal to initial capacity, then it is
        # unused. Count number of used bins.
        num_bins.append((bins_packed != capacity).sum())
    # Score of heuristic function is negative of average number of bins used
    # across instances (as we want to minimize number of bins).
    return -np.mean(num_bins)


@funsearch.evolve
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.

    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.

    Return:
        Array of same size as bins with priority score of each bin.
    """
    ratios = item / bins
    log_ratios = np.log(ratios)
    priorities = -log_ratios
    return priorities
'''

elif config_type == 'cap_set':
    
    additional_prompt = \
"""
Create an improved priority function for constructing 8-dimensional cap sets that demonstrates:
Novel vector priority strategy: Design a smarter vector selection strategy.
Parameter tuning points: Mark adjustable parameters using tunable([option1, option2, ...]) wrapper. Examples:
`if axis_balance_weight = tunable([0.1, 0.3, 0.5])`
`sorted(elements, key=lambda x: tunable([x.diversity, x.centrality]))`
Focus first on innovative vector selection heuristics, then expose tuning parameters via `tunable()`.

Note:
1. Do not generate the `tunable()` function implementation.
2. Any helper functions should be defined within the priority function.
"""

    specification = r'''
import numpy as np
import itertools
from typing import List, Tuple


@funsearch.run
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


@funsearch.evolve
def priority(el: tuple[int, ...]) -> float:
    """Returns the priority with which we want to add `el` to the cap set in `n=8` dimensions.
    
    Args:
        el: An 8-dimensional vector (tuple) with components in {0, 1, 2}.

    Return:
        Priority score determining selection order in greedy algorithm. Higher
        values indicate the vector should be considered earlier.
    """
    n = 8
    return 0.0
'''

elif config_type == 'cycle_graphs':

    additional_prompt = \
"""
I'm working on the maximum independent set problem in multi-dimensional cycle graphs using a greedy algorithm with a priority function to determine vertex selection order. Please help me develop a smarter `priority_v2` function by analyzing my reference implementations.

## What I Need
1. **BOLD EVOLUTION OF PRIORITY FUNCTION**: Please create a novel priority function variant that might outperform my reference implementations. Don't be constrained by my current approaches - take risks and suggest radically different strategies that might lead to breakthroughs.
2. **MARK ALL TUNABLE PARAMETERS**: For every single element that could potentially be tuned (no matter how minor), mark it with tunable([option1, option2, ...]) wrapper. 
  This includes but is not limited to:
    - Parameters and constants
    - Weighting factors
    - Thresholds
    - Logical conditions
    - Calculation methods
    - Function selection options
    - Algorithm hyperparameters
    - Anything else that might impact priority
  Format examples:
    - `if x == tunable([number_1, number_2, number_3])`
    - `sorted(elements, key=lambda x: tunable([x.property_1, x.property_2]))`

**My primary focus is on the conceptual innovation of the priority function itself.** While accurately marking tunable parameters is essential, please dedicate your main effort to designing the *core logic* of a potentially superior function first.

## Task Description
Please provide a Python function `priority_v2(el: tuple[int, ...], num_node: int, n: int) -> float` that:
1. Takes an n-dimensional vector (with elements in {0, 1, ..., m-1}) representing a vertex in the multi-dimensional torus graph.
2. Returns a priority score - higher scores indicate the vector should be considered earlier for addition to the indepentent set
3. Any helper functions and useful variables should be defined within the `priority_v2` function
## Current Priority Functions
Below are two or one reference priority functions I've developed and the num of vectexs they can select out.
"""

    specification = r'''
import itertools
import numpy as np


@funsearch.run
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
    scores = np.array([priority(tuple(child), num_nodes, n)
                        for child in children])

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


@funsearch.evolve
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
    return 0.
'''