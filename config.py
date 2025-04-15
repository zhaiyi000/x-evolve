import os


config_type = os.environ.get('CONFIG_TYPE', None)
if config_type not in ['bin_packing', 'cap_set', 'admissible_set']:
    raise Exception('wrong type')
n_dim = None
if config_type == 'cap_set':
    n_dim = os.environ.get('N_DIM', None)
    assert n_dim != None
    n_dim = int(n_dim)


log_dir = os.environ.get('LOG_DIR', 'logs')
sample_llm_cnt = 10


if config_type == 'bin_packing':
    evaluate_function_c_v1 = 0.1
    evaluate_function_c_l1 = 0.1
    evaluate_function_c_1 = 100
    evaluate_function_temperature = 0.1
    evaluate_function_mask_half = False

    sample_iterator_temperature = 1
    sample_iterator_no_update_cnt = 3

    sample_llm_api_min_score = -500

    measure_timeout = 15

elif config_type == 'cap_set':
    evaluate_function_c_v1 = 0.
    evaluate_function_c_l1 = 0.
    evaluate_function_c_1 = 1
    evaluate_function_temperature = 10
    evaluate_function_mask_half = True

    sample_iterator_temperature = 100
    sample_iterator_no_update_cnt = 5

    sample_llm_api_min_score = 256

    measure_timeout = 15


elif config_type == 'admissible_set':
    evaluate_function_c_v1 = 0.
    evaluate_function_c_l1 = 0.
    evaluate_function_c_1 = 1
    evaluate_function_temperature = 10
    evaluate_function_mask_half = True

    sample_iterator_temperature = 100
    sample_iterator_no_update_cnt = 1

    sample_llm_api_min_score = 548

    measure_timeout = 60

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

    specification = r'''import numpy as np


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
f'''I'm working on the {n_dim}-dimensional cap set problem using a greedy algorithm with a priority function to determine vector selection order. Please help me develop a smarter `priority_v2` function by analyzing my reference implementations.


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
Please provide a Python function `priority_v2(el: tuple[int, ...]) -> float` that:
1. Takes an {n_dim}-dimensional vector (with elements in {{0,1,2}})
2. Returns a priority score - higher scores indicate the vector should be considered earlier for addition to the Cap Set
3. Any helper functions should be defined within the `priority_v2` function


## Current Priority Functions
Below are two reference priority functions I've developed.
'''

    specification = f'''import numpy as np
import itertools
import math


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
    """Returns the priority with which we want to add `el` to the cap set in `n={n_dim}` dimensions.
    
    Args:
        el: An {n_dim}-dimensional vector (tuple) with components in {{0, 1, 2}}.

    Return:
        Priority score determining selection order in greedy algorithm. Higher
        values indicate the vector should be considered earlier.
    """
    n = {n_dim}
    return 0.0
'''


elif config_type == 'admissible_set':
    
    additional_prompt = \
'''I'm working on the constant-weight admissible set problem with dimension 12 and weight 7, using a greedy algorithm that relies on a priority function to determine the vector selection order. Please help me develop a smarter `priority_v2` function by analyzing my reference implementations.


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
Please provide a Python function `priority_v2(el: np.ndarray) -> float` that:
1. Takes an 12-dimensional vector (with elements in {0,1,2})
2. Returns a priority score - higher scores indicate the vector should be considered earlier for addition to the admissible set
3. **Use NumPy vectorized operations as much as possible**
4. Any helper functions should be defined within the `priority_v2` function


## Current Priority Functions
Below are two reference priority functions I've developed.
'''

    specification = r'''import itertools
import math
import numpy as np


def solve(n: int, w: int) -> np.ndarray:
    """Generates a constant-weight admissible set I(n, w)."""
    import block_cpp
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
        block_cpp.block_children(scores, max_admissible_set, child)
        max_admissible_set = np.concatenate([max_admissible_set, child[None]], axis=0)

    return max_admissible_set


@funsearch.run
def evaluate(kargs) -> int:
    """Returns the size of the constructed admissible set."""
    return len(solve(kargs['n'], kargs['w']))


@funsearch.evolve
def priority(el: np.ndarray) -> float:
    """Computes a priority score for an element to determine its order of addition to the admissible set.
    
    Args:
        el: A numpy array representing an element with n=12 positions, where each position contains 0, 1, or 2. Elements have weight w=7 (meaning 7 non-zero values).

    Return:
        A float score where higher values indicate higher priority for inclusion in the admissible set.
    """
    n = 12
    w = 7
    return 0
'''