import numba
import numpy as np
import random
import copy
import traceback

def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero(bins - item >= 0)[0]

def online_binpack(items: tuple[float, ...], bins: np.ndarray) -> tuple[list[list[float, ...], ...], np.ndarray]:
    """Performs online binpacking of `items` into `bins`."""
    packing = [[] for _ in bins]
    for item in items:
        valid_bin_indices = get_valid_bin_indices(item, bins)
        priorities = priority(item, bins[valid_bin_indices])
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)
    packing = [bin_items for bin_items in packing if bin_items]
    return (packing, bins)

def evaluate(instances: dict) -> float:
    """Evaluate heuristic function on a set of online binpacking instances."""
    num_bins = []
    for name in instances:
        instance = instances[name]
        capacity = instance['capacity']
        items = instance['items']
        bins = np.array([capacity for _ in range(instance['num_items'])])
        (_, bins_packed) = online_binpack(items, bins)
        num_bins.append((bins_packed != capacity).sum())
    return -np.mean(num_bins)

@numba.jit(nopython=True)
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.

    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.

    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    # Assign higher priority to bins with smaller remaining space.
    # Use a small tie-breaker to prefer bins that were more filled before adding the item.
    return -remaining - (bins / 1e9)

    




import bin_packing_utils
bin_packing_or3 = {'OR3': bin_packing_utils.datasets['OR3']}
result = evaluate(bin_packing_or3['OR3'])
print(result)


# cat run_api.log | grep 'Score        :' | grep -v 'None' | awk '{print $3}' | sort -nr | head -n 1
# 207.45  funsearch
# 201.2   best
# -212.75   base
# 208.35   now best 
