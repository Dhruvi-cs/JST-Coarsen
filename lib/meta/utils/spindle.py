R"""
"""
#
import numpy as onp
import numpy.typing as onpt
import itertools
import more_itertools as xitertools
from typing import List

def keep_order_in_A(A, B):
    """
    Efficiently keep elements from A that appear in B, preserving A order.
    Uses np.isin with optimal kind selection for best performance.
    """
    if isinstance(A, list):
        A = onp.array(A)
    if isinstance(B, list):
        B = onp.array(B)
        
    # Use kind='table' for integers (faster, lower memory than default)
    if onp.issubdtype(B.dtype, onp.integer):
        mask = onp.isin(A, B, kind='table')
    else:
        print("Warning: Using default kind for np.isin; performance may be suboptimal.")
        mask = onp.isin(A, B)
    return A[mask]

def fitsplit(
    indices: onpt.NDArray[onp.generic], 
    proportion: List[int],
    priority: List[int]
) -> List[onpt.NDArray[onp.generic]]:
    R"""
    Split given indices (sorted by node in-degree large to small) 
    fitting to given proportion as close as possible with
    minor adjustment based on priority.
    """

    # Safety check.
    if onp.gcd.reduce(proportion) != 1:
        # EXPECT:
        # It is possible to have improper arguments.
        raise RuntimeError(
            "Proportion of all splits should have greatest common divisor "
            "being 1 to work properly."
        )
    if len(onp.unique(priority)) != len(priority):
        # It is possible to have improper arguments.
        raise RuntimeError(
            "Priority of all splits should be unique to work properly."
        )

    # Break raw indices into chunks fitting the split proportion unit.
    # Get the unit chunk coverage of all splits.
    chunkunits = xitertools.distribute(sum(proportion), indices)
    ties = list(itertools.accumulate(xitertools.prepend(0, proportion)))
    #
    splits: List[List[int]]

    # Fill each split with its covered unit chunks in order of priority.
    splits = [[] for _ in proportion]
    for (sp, begin, end) in (
        zip(
            xitertools.sort_together(
                (priority, range(len(proportion))),
                reverse=True,
            )[1],
            ties[:-1], ties[1:],
        )
    ):
        # Use roundrobin to concatenate sample chains of the same chunk to keep
        # the priority ordering in the chunk.
        splits[sp].extend(xitertools.roundrobin(*chunkunits[begin:end]))
    return [onp.array(split) for split in splits]




def stratified_fitsplit2(
    indices: onpt.NDArray[onp.generic], 
    labels: onpt.NDArray[onp.generic],
    proportion: List[int],
    priority: List[int],
) -> List[onpt.NDArray[onp.generic]]:
    R"""
    Split given indices fitting to given proportion while maintaining label
    distribution across splits. Each label class is split proportionally and
    then combined.
    
    Args:
        indices: Array of indices to split
        labels: Array of labels corresponding to indices
        proportion: List of integers representing split proportions
        priority: List of integers representing priority order for splits
        
    Returns:
        List of arrays containing split indices
    """
    # Safety check.
    if onp.gcd.reduce(proportion) != 1:
        raise RuntimeError(
            "Proportion of all splits should have greatest common divisor "
            "being 1 to work properly."
        )
    if len(onp.unique(priority)) != len(priority):
        raise RuntimeError(
            "Priority of all splits should be unique to work properly."
        )
    if len(indices) != len(labels):
        raise RuntimeError(
            "Length of indices and labels must match."
        )
    # 
    
    if labels.ndim > 2:
        # one-hot encoded labels
        labels = onp.argmax(labels, axis=1)
    
    labels = labels.flatten()
    unique_labels = onp.unique(labels)
    
    # Initialize split containers
    splits = [[] for _ in proportion]
    
    
    # For each label class, split proportionally
    for label in unique_labels:
        # Get indices for this label
        label_mask = labels == label
        label_indices = indices[label_mask]
        
        # Split this label's indices using fitsplit
        label_splits = fitsplit(
            label_indices, 
            proportion=proportion,
            priority=priority
        )
        
        # Add to corresponding splits
        for i, label_split in enumerate(label_splits):
            splits[i].extend(label_split)
    
    # Convert to numpy arrays and shuffle each split to mix labels
    rng = onp.random.default_rng(42)
    result_splits = []
    for split in splits:
        split_array = onp.array(split)
        rng.shuffle(split_array)
        result_splits.append(split_array)
    
    return result_splits


from sklearn.model_selection import StratifiedShuffleSplit

def stratified_fitsplit(
    indices: onpt.NDArray[onp.generic], 
    labels: onpt.NDArray[onp.generic],
    proportion: List[int],
    priority: List[int],
) -> List[onpt.NDArray[onp.generic]]:
    R"""
    Split given indices fitting to given proportion while maintaining label
    distribution across splits. Each label class is split proportionally and
    then combined.
    
    Args:
        indices: Array of indices to split
        labels: Array of labels corresponding to indices
        proportion: List of integers representing split proportions
        priority: List of integers representing priority order for splits
        
    Returns:
        List of arrays containing split indices
    """
    FIXED_RANDOM_SEED = 42
    
    if labels.ndim > 2:
        # one-hot encoded labels
        labels = onp.argmax(labels, axis=1)
    
    labels = labels.flatten()
    unique_labels = onp.unique(labels)
    
    # use stratified split from sklearn
    if len(proportion) == 2:
        # train/val split
        sss = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=proportion[1]/sum(proportion),
            random_state=FIXED_RANDOM_SEED
        )
        train_idx, val_idx = next(sss.split(indices, labels))
        return [indices[train_idx], indices[val_idx], []]
    elif len(proportion) == 3:
        # train/val/test split
        sss1 = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=proportion[2]/sum(proportion),
            random_state=FIXED_RANDOM_SEED
        )
        train_val_idx, test_idx = next(sss1.split(indices, labels))
        
        sss2 = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=proportion[1]/(proportion[0]+proportion[1]),
            random_state=FIXED_RANDOM_SEED
        )
        train_idx, val_idx = next(sss2.split(indices[train_val_idx], labels[train_val_idx]))
        
        return [indices[train_val_idx][train_idx], indices[train_val_idx][val_idx], indices[test_idx]]
    else:
        raise NotImplementedError(
            "Stratified fitsplit currently supports only 2 or 3 splits."
        )