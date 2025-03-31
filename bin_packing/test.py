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

def tunable(li):
    return li[0]


ls_i = 0
def tunable(ls):
    ls = ['0.2', '0.8', '4.0', '0.1', '0.5', '0.2', '0.9', '0.1', '0.5', '0.2', '0.8', '0.8', '0.6', '0.6', '0.2', '0.8', '0.3', '0.4', '0.5', '0.1', '0.4']
    global ls_i
    item = float(ls[ls_i])
    ls_i += 1
    return item



# def priority(item: float, bins: np.ndarray) -> np.ndarray:
#     global ls_i
#     ls_i = 0
#     """Returns priority with which we want to add item to each bin.

#     Args:
#         item: Size of item to be added to the bin.
#         bins: Array of capacities for each bin.

#     Return:
#         Array of same size as bins with priority score of each bin.
#     """
#     """Advanced priority function for online bin-packing with tunable parameters."""
#     # Calculate spatial fit metrics with tunable bounds
#     spatial_fit = np.zeros_like(bins, dtype=np.float64)
#     valid_bins = bins >= item
    
#     # Dynamic utilization bands based on item size relative to average bin size
#     avg_bin_size = np.mean(bins)
#     utilization = item / bins
    
#     # Tunable parameters for dynamic bounds
#     lower_bound = tunable([0.1, 0.2, 0.3]) * (item / avg_bin_size)
#     upper_bound = tunable([0.7, 0.8, 0.9]) * (item / avg_bin_size)
    
#     # Clip utilization within dynamic bounds
#     utilization_clipped = np.clip(utilization, lower_bound, upper_bound)
    
#     # Spatial fit calculation with adaptive exponent
#     spatial_fit_exponent = tunable([2.0, 3.0, 4.0]) +\
#                          tunable([0.1, 0.2, 0.3]) * (1 - np.mean(valid_bins))
#     spatial_fit[valid_bins] = np.power(1.0 / utilization_clipped[valid_bins], spatial_fit_exponent)
    
#     # Calculate remaining capacity and future potential
#     remaining_capacity = bins - item
#     remaining_capacity[~valid_bins] = -1  # Mark invalid bins
    
#     # Future potential with tunable scaling
#     future_potential = np.zeros_like(bins, dtype=np.float64)
#     valid_remaining = remaining_capacity > 0
    
#     # Dynamic scaling factors with tunable weights
#     utilization_factor = tunable([0.5, 0.6, 0.7]) + (1 - np.mean(valid_bins)) * tunable([0.1, 0.2, 0.3])
#     size_factor = tunable([0.8, 0.9, 1.0]) * (item / np.max(bins))
    
#     future_potential[valid_remaining] = (
#         np.log(remaining_capacity[valid_remaining] + 1e-6) * 
#         utilization_factor * 
#         size_factor
#     )
    
#     # Bin flexibility metric with tunable thresholds
#     flexibility_score = np.zeros_like(bins, dtype=np.float64)
#     low_threshold = tunable([0.1, 0.2, 0.3]) * bins
#     high_threshold = tunable([0.3, 0.4, 0.5]) * bins
    
#     flexibility_score[valid_bins] = (
#         1.0 - np.abs((remaining_capacity[valid_bins] - tunable([0.2, 0.25, 0.3]) * bins[valid_bins]) / bins[valid_bins])
#     )
#     flexibility_score[remaining_capacity < low_threshold] *= tunable([0.8, 0.9, 1.0])
#     flexibility_score[remaining_capacity > high_threshold] *= tunable([0.7, 0.8, 0.9])
    
#     # Order-aware priority adjustment with tunable weight
#     order_factor = tunable([0.5, 0.6, 0.7]) * np.arange(len(bins)) / len(bins)
#     priority_adjustment = order_factor * flexibility_score
    
#     # Dynamic weight adjustment with tunable parameters
#     current_fit_weight = tunable([0.4, 0.5, 0.6]) + tunable([0.1, 0.2, 0.3]) * (1 - np.mean(valid_bins))
#     future_weight = tunable([0.6, 0.7, 0.8]) + tunable([0.1, 0.2, 0.3]) * np.mean(valid_bins)
#     flexibility_weight = tunable([0.3, 0.4, 0.5])
    
#     # Combine metrics with dynamic weights
#     combined_score = (
#         current_fit_weight * spatial_fit +
#         future_weight * future_potential +
#         flexibility_weight * flexibility_score +
#         priority_adjustment
#     )
    
#     # Apply penalties for extreme utilization with tunable thresholds
#     penalty_factor = tunable([0.5, 0.6, 0.7])
#     penalties = np.zeros_like(bins, dtype=np.float64)
#     penalties[remaining_capacity < tunable([0.1, 0.2, 0.3]) * bins] += penalty_factor
#     penalties[remaining_capacity > tunable([0.4, 0.5, 0.6]) * bins] += penalty_factor
    
#     combined_score -= penalties
    
#     # Normalize the scores
#     max_score = np.max(combined_score)
#     if max_score != 0:
#         combined_score /= max_score
    
#     # Higher priority means prefer to place the item in this bin
#     priorities = -combined_score  # Negative for proper sorting (lower values first)
    
#     return priorities


def priority(item: float, bins: np.ndarray) -> np.ndarray:
  """Heursitic discovered for the OR datasets."""
  def s(bin, item):
    if bin - item <= 2:
      return 4
    elif (bin - item) <= 3:
      return 3
    elif (bin - item) <= 5:
      return 2
    elif (bin - item) <= 7:
      return 1
    elif (bin - item) <= 9:
      return 0.9
    elif (bin - item) <= 12:
      return 0.95
    elif (bin - item) <= 15:
      return 0.97
    elif (bin - item) <= 18:
      return 0.98
    elif (bin - item) <= 20:
      return 0.98
    elif (bin - item) <= 21:
      return 0.98
    else:
      return 0.99

  return np.array([s(b, item) for b in bins])


# import bin_packing_utils
from parse_binpack import datasets


# cat run_api_v4.log | grep 'Score        :' | grep -v 'None' | awk '{print $3}' | sort -nr | head -n 1
# 212.0
# 207.45
# 201.2
# deepseek 340.1 -212.4


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


input_names = ['OR1', 'OR2', 'OR3', 'OR4']
for input_name in input_names:

    print(evaluate(datasets[input_name]))
    print(l1_bound_dataset(datasets[input_name]))



# -52.25
# 49.05
# -106.3
# 101.55
# -207.15
# 201.2
# -409.5
# 400.55


# -51.65
# 49.05
# -105.8
# 101.55
# -207.45
# 201.2
# -410.45
# 400.55
