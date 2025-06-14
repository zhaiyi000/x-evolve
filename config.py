
# import debugpy
# debugpy.listen(5678)
# print('wait_for_client...')
# debugpy.wait_for_client()
# debugpy.breakpoint()


import os


config_type = os.environ.get('CONFIG_TYPE', None)
if config_type not in ['bin_packing', 'cap_set', 'admissible_set', 'symmetry_admissible_set', 'cycle_graphs', 'corners']:
    raise Exception('wrong type')


n_dim = None
w_dim = None
n_w_dim = None
nodes_dim = None
num_nodes = None

if config_type == 'cap_set' or config_type == 'corners':
    n_dim = os.environ.get('N_DIM', None)
    assert n_dim != None
    n_dim = int(n_dim)
elif config_type == 'symmetry_admissible_set':
    n_w_dim = os.environ.get('N_W_DIM', None)
    assert n_w_dim in ['15_10', '21_15', '24_17', '27_19']
    if n_w_dim == '15_10':
        n_dim = 15
        w_dim = 10
    elif n_w_dim == '21_15':
        n_dim = 21
        w_dim = 15
    elif n_w_dim == '24_17':
        n_dim = 24
        w_dim = 17
    elif n_w_dim == '27_19':
        n_dim = 27
        w_dim = 19
    else:
        raise Exception('wrong type')
elif config_type == 'cycle_graphs':
    nodes_dim = os.environ.get('NODES_DIM', None)
    assert nodes_dim != None 
    num_nodes, n_dim = map(int, nodes_dim.split('_'))
    

log_dir = os.environ.get('LOG_DIR', 'logs')
sample_llm_cnt = 10
update_database_cnt = 3
island_cnt = 4


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
    # evaluate_function_c_v1 = 0.
    # evaluate_function_c_l1 = 0.
    # evaluate_function_c_1 = 1
    # evaluate_function_temperature = 10
    # evaluate_function_mask_half = True

    sample_iterator_temperature = 100
    sample_iterator_no_update_cnt = 3

    if n_dim == 7:
        sample_llm_api_min_score = 128
    elif n_dim == 8:
        sample_llm_api_min_score = 256
    elif n_dim == 9:
        sample_llm_api_min_score = 512
    else:
        raise Exception('wrong n dim')

    measure_timeout = 30


elif config_type == 'admissible_set':
    # evaluate_function_c_v1 = 0.
    # evaluate_function_c_l1 = 0.
    # evaluate_function_c_1 = 1
    # evaluate_function_temperature = 10
    # evaluate_function_mask_half = True

    sample_iterator_temperature = 100
    sample_iterator_no_update_cnt = 1

    sample_llm_api_min_score = 548

    measure_timeout = 15

elif config_type == 'symmetry_admissible_set':
    # evaluate_function_c_v1 = 0.
    # evaluate_function_c_l1 = 0.
    # evaluate_function_c_1 = 1
    # evaluate_function_temperature = 10
    # evaluate_function_mask_half = True

    sample_iterator_no_update_cnt = 1

    sample_llm_api_min_score = 548

    if n_w_dim == '15_10':
        measure_timeout = 30
        sample_iterator_temperature = 1000
    elif n_w_dim == '21_15':
        measure_timeout = 30
        sample_iterator_temperature = 10000
    elif n_w_dim == '24_17':
        measure_timeout = 300
        sample_iterator_temperature = 100000
    elif n_w_dim == '27_19':
        measure_timeout = 1800
        sample_iterator_temperature = 200000
    else:
        raise Exception('wrong n w dim')

elif config_type == 'cycle_graphs':
    # evaluate_function_c_v1 = 0.1
    # evaluate_function_c_l1 = 0.1
    # evaluate_function_c_1 = 10
    # evaluate_function_temperature = 1
    # evaluate_function_mask_half = True

    sample_iterator_temperature = 100
    sample_iterator_no_update_cnt = 1

    sample_llm_api_min_score = 243

    measure_timeout = 30

elif config_type == 'corners':
    # evaluate_function_c_v1 = 0.1
    # evaluate_function_c_l1 = 0.1
    # evaluate_function_c_1 = 10
    # evaluate_function_temperature = 1
    # evaluate_function_mask_half = True

    sample_iterator_temperature = 50
    sample_iterator_no_update_cnt = 3

    sample_llm_api_min_score = 0

    measure_timeout = 15

else:
    raise Exception('wrong type')



if config_type == 'bin_packing':
    
    additional_prompt = \
"""I'm working on the online bin-packing problem using a greedy algorithm with a priority function.


## What I Need
1. **BOLD EVOLUTION OF PRIORITY FUNCTION**: Please create an improved `priority_new` function that might outperform my reference implementations. Don't be constrained by my current approaches - take risks and suggest radically different strategies that might lead to breakthroughs.
2. **MARK ALL TUNABLE PARAMETERS**: For every element in the `priority_new` function that could potentially be tuned, wrap it with tunable([option1, option2, ...]).
  Format examples:
    - `if x == tunable([x1, x2, x3]):`
    - `z = tunable([x + y, x * (y + 1)])`


## Task Description
Please help me develop an improved `priority_new` function by analyzing my reference implementations.
Output Python code only, without any comments.
Keep it as short as possible.
Don't use random functions.


## Current Priority Functions
Below are two reference priority functions I've developed.
"""
# Create an improved Python function for online bin-packing that demonstrates:
# Novel priority strategy: Propose a smarter item-bin matching approach considering both spatial fit and future packing potential.
# Parameter tuning points: Clearly mark tuning parameters using tunable([option1, option2, ...]) wrapper. Examples:
# `if remaining_capacity > tunable([0.2, 0.5]):`
# `sorted(items, key=lambda x: tunable([x.size, x.weight]))`
# Focus first on strategic innovation, then expose tuning parameters through tunable([option1, option2, ...]) calls. Keep implementation practical but non-trivial.

    specification = f'''import numpy as np


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
    return 0
'''

elif config_type == 'cap_set':
    
    additional_prompt = \
f'''I'm working on the {n_dim}-dimensional cap set problem using a greedy algorithm with a priority function to determine vector selection order. A cap set is a collection of vectors in {{0,1,2}}^n where no three vectors form an arithmetic line (i.e., for any three distinct vectors a, b, c, if a + c = 2b, then they cannot all appear in the cap set).


## What I Need
1. **BOLD EVOLUTION OF PRIORITY FUNCTION**: Please create an improved `priority_new` function that might outperform my reference implementations. Don't be constrained by my current approaches - take risks and suggest radically different strategies that might lead to breakthroughs.
2. **MARK ALL TUNABLE PARAMETERS**: For every element in the `priority_new` function that could potentially be tuned, wrap it with tunable([option1, option2, ...]).
  Format examples:
    - `if x == tunable([x1, x2, x3]):`
    - `z = tunable([x + y, x * (y + 1)])`


## Task Description
Please help me develop an improved `priority_new` function by analyzing my reference implementations.
Output Python code only, without any comments.
The score is computed based on the relationships among el[i], el[-i], el[i - k], el[i + k], and el[(i + k) % n].


## Current Priority Functions
Below are two reference priority functions I've developed.
'''

    specification = f'''import numpy as np
import itertools


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

elif config_type == 'cycle_graphs':

    additional_prompt = \
f'''I'm working on the maximum independent set problem in the {n_dim}-th strong product of a {num_nodes}-node cycle graph, using a greedy algorithm guided by a priority function to determine vector selection order.Each vertex in this graph is a {n_dim}-dimensional vector with values in {0, 1, ..., num_nodes-1}. Two vertices are adjacent if they differ by at most 1 (mod {num_nodes}) in each coordinate and are not identical. An independent set is a subset of vectors where no two are adjacent.


## What I Need
1. **BOLD EVOLUTION OF PRIORITY FUNCTION**: Please create an improved `priority_new` function that might outperform my reference implementations. Don't be constrained by my current approaches - take risks and suggest radically different strategies that might lead to breakthroughs.
2. **MARK ALL TUNABLE PARAMETERS**: For every element in the `priority_new` function that could potentially be tuned, wrap it with tunable([option1, option2, ...]).
  Format examples:
    - `if x == tunable([x1, x2, x3]):`
    - `z = tunable([x + y, x * (y + 1)])`


## Task Description
Please help me develop an improved `priority_new` function by analyzing my reference implementations.
Output Python code only, without any comments.
The function should be concise.


## Current Priority Functions
Below are two reference priority functions I've developed.
'''

    specification = f'''
import itertools
import numpy as np
import pickle

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
    with open('cycle_graphs{nodes_dim}.pkl', 'rb') as f:
        to_block, powers, children = pickle.load(f)
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
    num_nodes = {num_nodes}
    n = {n_dim}
    return 0.
'''

elif config_type == 'admissible_set':
    
    additional_prompt = \
'''I'm working on the constant-weight admissible set problem with dimension 12 and weight 7, using a greedy algorithm that relies on a priority function to determine the vector selection order.


## What I Need
1. **BOLD EVOLUTION OF PRIORITY FUNCTION**: Please create a novel `priority_v2` function that might outperform my reference implementations. Don't be constrained by my current approaches - take risks and suggest radically different strategies that might lead to breakthroughs.
2. **MARK ALL TUNABLE PARAMETERS**: For every element in the `priority_v2` function that could potentially be tuned, wrap it with tunable([option1, option2, ...]).
  Format examples:
    - `if x == tunable([num_1, num_2, num_3])`
    - `y = tunable([np.exp(x), np.log(x)))`


## Task Description
Please help me develop a smarter `priority_v2` function by analyzing my reference implementations.
1. Keep the exact function signature: `def priority_v2(el: tuple[int, ...]) -> float:`.
2. Output only Python code, without imports, helper functions, or comments. Keep it as short and simple as possible.
3. Use only basic logical rules, such as position, symmetry, and element presence, while avoiding complex mathematical modeling (including statistical calculations).


## Current Priority Functions
Below are two reference priority functions I've developed.
'''

    specification = '''import itertools
import numpy as np


def solve(n: int, w: int) -> np.ndarray:
    """Generates a constant-weight admissible set I(n, w)."""
    import cpp_helper
    import pickle
    with open('admissible_set_scores.pkl', 'rb') as f:
        children, scores = pickle.load(f)
    for child_index, child in enumerate(children):
        if scores[child_index] == 0:
            scores[child_index] = priority(child.tolist())

    max_admissible_set = np.empty((0, n), dtype=np.int32)
    while np.any(scores != -np.inf):
        # Find element with largest score.
        max_index = np.argmax(scores)
        child = children[max_index]
        # block_children(scores, max_admissible_set, child)
        cpp_helper.block_children(scores, max_admissible_set, child)
        max_admissible_set = np.concatenate([max_admissible_set, child[None]], axis=0)

    return max_admissible_set


@funsearch.run
def evaluate(kargs) -> int:
    """Returns the size of the constructed admissible set."""
    return len(solve(kargs['n'], kargs['w']))


@funsearch.evolve
def priority(el: tuple[int, ...]) -> float:
    """Computes a priority score for an element to determine its order of addition to the admissible set.
    
    Args:
        el: A tuple representing a vector with n=12 positions, where each position can be 0, 1, or 2. The element has a weight w=7, meaning it contains 7 non-zero values.

    Return:
        A float score where higher values indicate higher priority for inclusion in the admissible set.
    """
    n = 12
    w = 7
    return 0
'''


elif config_type == 'symmetry_admissible_set':
    
    additional_prompt = \
f'''I'm working on the constant-weight admissible set problem with dimension {n_dim} and weight {w_dim}, using a greedy algorithm that relies on a priority function to determine the vector selection order.


## What I Need
1. **BOLD EVOLUTION OF PRIORITY FUNCTION**: Please create an improved `priority_new` function that might outperform my reference implementations. Don't be constrained by my current approaches - take risks and suggest radically different strategies that might lead to breakthroughs.
2. **MARK ALL TUNABLE PARAMETERS**: For every element in the `priority_new` function that could potentially be tuned, wrap it with tunable([option1, option2, ...]).
  Format examples:
    - `if x == tunable([x1, x2, x3]):`
    - `z = tunable([x + y, x * (y + 1)])`


## Task Description
Please help me develop an improved `priority_new` function by analyzing my reference implementations.
Output Python code only, without any comments.
The score is computed based on the relationships among el[i], el[-i], el[i - k], el[i + k], and el[(i + k) % n].


## Current Priority Functions
Below are two reference priority functions I've developed.
'''

    specification = f'''import itertools
import numpy as np


def expand_admissible_set(
    pre_admissible_set: list[tuple[int, ...]],
    TRIPLES
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

def solve(n: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Generates a symmetric constant-weight admissible set I(n, w)."""
    
    TRIPLES = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 2), (0, 2, 1), (1, 1, 1), (2, 2, 2)]
    INT_TO_WEIGHT = [0, 1, 1, 2, 2, 3, 3]
    
    num_groups = n // 3
    assert 3 * num_groups == n
    import cpp_helper

    # # Compute the scores of all valid (weight w) children.
    # valid_children = []
    # for child in itertools.product(range(7), repeat=num_groups):
    #     weight = sum(INT_TO_WEIGHT[x] for x in child)
    #     if weight == w:
    #         valid_children.append(np.array(child, dtype=np.int32))

    valid_children = np.load('admissible_set_{n_w_dim}.npy')
    valid_children_expand = np.load('admissible_set_{n_w_dim}_expand.npy')
    valid_children_expand = [tuple(xs) for xs in valid_children_expand.tolist()]

    valid_scores = np.array(
        [priority(xs) for xs in valid_children_expand]
    )

    pre_admissible_set = cpp_helper.greedy_search(
        num_groups, valid_scores, valid_children
    )
    return pre_admissible_set, np.array(expand_admissible_set(pre_admissible_set, TRIPLES))


@funsearch.run
def evaluate(kargs) -> int:
    """Returns the size of the expanded admissible set."""
    _, admissible_set = solve(kargs['n'], kargs['w'])
    return len(admissible_set)


@funsearch.evolve
def priority(el: tuple[int, ...]) -> float:
    """Computes a priority score for an element to determine its order of addition to the admissible set.
    
    Args:
        el: A tuple representing a vector with n={n_dim} positions, where each position can be 0, 1, or 2. The element has a weight w={w_dim}, meaning it contains {w_dim} non-zero values.

    Return:
        A float score where higher values indicate higher priority for inclusion in the admissible set.
    """
    n = {n_dim}
    w = {w_dim}
    return 0
'''

    mask_skip_line_cnt = 2


elif config_type == 'corners':
    raise Exception('todo')

    additional_prompt = \
f'''I'm working on the {n_dim}-dimensional corners problem using a greedy algorithm with a priority function to determine vector selection order. A corner-free set is a subset of vectors in \((\mathbbZ_2^{n_dim} times \mathbbZ_2^{n_dim})\) such that no three vectors form a corner — that is, no triple of the form \((x, y), (x+\lambda, y), (x, y+\lambda)\), with \(\lambda != 0\), all appear in the set (arithmetic done modulo 2).

## What I Need
1. **BOLD EVOLUTION OF PRIORITY FUNCTION**: Please create an improved `priority_v2` function that might outperform my reference implementations. Don't be constrained by my current approaches - take risks and suggest radically different strategies that might lead to breakthroughs.
2. **MARK ALL TUNABLE PARAMETERS**: For every element in the `priority_v2` function that could potentially be tuned, wrap it with tunable([option1, option2, ...]).
  Format examples:
    - `if x == tunable([x1, x2, x3]):`
    - `z = tunable([x + y, x * (y + 1))`


## Task Description
Please help me develop an improved `priority_v2` function by analyzing my reference implementations.
1. Keep the exact function signature: `def priority_v2(el: tuple[int, ...], n: int) -> float:`.
2. Output only Python code, without imports, helper functions, or comments. Keep it as short and simple as possible.
3. Use a basic heuristic approach; avoid complex statistical methods.
4. **Don't use corners**


## Current Priority Functions
Below are two reference priority functions I've developed.
'''


    specification = f'''import corners
import itertools
import numpy as np


@funsearch.run
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

@funsearch.evolve
def priority(el: tuple[int, ...], n: int)-> float:
    """Returns the priority with which we want to add `el`.
    Args:
        el: A candidate element to be considered, as a tuple of length `2 * n` with
        elements in {0, 1}.
        n: Power of the graph.
    Returns:
        A number reflecting the priority with which we want to add `el` to the set.
    """
    return 0.
'''

