import numpy as np


def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]


def online_binpack(items: tuple[float, ...], bins: np.ndarray) -> tuple[list[list[float, ...], ...], np.ndarray]:
    """Performs online binpacking of `items` into `bins`."""
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


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.

    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.

    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    mask = remaining >= 0
    fill_ratio = (item / bins) ** 4.0
    penalty = np.where(~mask, -500.0 * (remaining ** 1.0), 0)
    base_score = 0.4 * (1 - 0.6 * remaining / bins.max())
    score = np.where(mask, base_score, base_score + penalty)
    threshold1 = 0.05 * bins.max()
    threshold2 = 0.01 * bins.max()
    score = np.where(remaining < threshold1,
                     0.5 * base_score + 0.3 * fill_ratio,
                     base_score + -0.7 * fill_ratio)
    score = np.where(remaining < threshold2, 50, score)
    score = np.where(remaining / bins < 0.15, score * 4.0, score)
    return score

# 5.50% 4.43% 2.98% 2.45%