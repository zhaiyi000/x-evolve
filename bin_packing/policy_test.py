import numpy as np
from parse_binpack import datasets
import importlib

def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]

def is_valid_packing(
    packing: list[list[float, ...], ...], items: list[float], capacity: float
) -> bool:
    """Returns whether `packing` is valid.

    Returns whether `packing` is a valid packing of `items` into bins of size
    `capacity`.

    Args:
        packing: Packing of items into bins. List of bins, where each bin contains
        a list of items packed into that bin.
        items: List of item sizes.
        capacity: Capacity of each bin.
    """
    # Check that items in packing are exactly the same as list of input items.
    packed_items = sum(packing, [])  # Join items in each bin into a single list.
    if sorted(packed_items) != sorted(items):
        return False

    # Check that each bin contains less than `capacity` items .
    for bin_items in packing:
        if sum(bin_items) > capacity:
            return False
    return True

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
        packing, bins_packed = online_binpack(items, bins)
        # If remaining capacity in a bin is equal to initial capacity, then it is
        # unused. Count number of used bins.
        if is_valid_packing(packing, items, capacity):
            num_bins.append((bins_packed != capacity).sum())
            # Score of heuristic function is negative of average number of bins used
            # across instances (as we want to minimize number of bins).
        else:
            return float('-inf')
    return -np.mean(num_bins)


def l1_bound(items: tuple[int, ...], capacity: int) -> float:
    """Computes L1 lower bound on OPT for bin packing.

    Args:
        items: Tuple of items to pack into bins.
        capacity: Capacity of bins.

    Returns:
        Lower bound on number of bins required to pack items.
    """
    return np.ceil(np.sum(items) / capacity)


def l1_bound_dataset(instances: dict) -> float:
    """Computes the mean L1 lower bound across a dataset of bin packing instances.

    Args:
        instances: Dictionary containing a set of bin packing instances.

    Returns:
        Average L1 lower bound on number of bins required to pack items.
    """
    l1_bounds = []
    for name in instances:
        instance = instances[name]
        l1_bounds.append(l1_bound(instance['items'], instance['capacity']))
    return np.mean(l1_bounds)


for i in range(10):
    module_name = f"OR_newprompt.program_{i}"
    module = importlib.import_module(module_name)
    priority = module.priority
    
    print(f'program_{i}:')

    opt_num_bins = {}
    for name, dataset in datasets.items():
        opt_num_bins[name] = l1_bound_dataset(dataset)
    
    input_names = ['OR1', 'OR2', 'OR3', 'OR4']
    score_list = []
    for input_name in input_names:
        score = evaluate(datasets[input_name])
        print(score)
        score_list.append(score)
        avg_num_bins = -score
        excess = (avg_num_bins - opt_num_bins[input_name]) / opt_num_bins[input_name]
        print(f'\t Average number of bins: {avg_num_bins}')
        print(f'\t Lower bound on optimum: {opt_num_bins[input_name]}')
        print(f'\t Excess: {100 * excess:.2f}%')
        print()
