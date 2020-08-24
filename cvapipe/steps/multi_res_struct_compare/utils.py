import numpy as np
from scipy.stats import pearsonr, gmean
from skimage.measure import block_reduce


def blockreduce_pyramid(input_arr, block_size=(2, 2, 2), func=np.max, max_iters=12):
    """
    Parameters
        ----------
        input_arr: np.array
            Input array to iteratively downsample
            Default: Path("local_staging/singlecellimages/manifest.csv")
        block_size: Tuple(int)
            Block size for iterative array reduction.  All voxels in this block
            are merged via func into one voxel during the downsample.
            Default: (2, 2, 2)
        func: Callable[[np.array], float]
            Function to apply to block_size voxels to merge them into one new voxel.
            Default: np.max
        max_iters: int
            Maximum number of downsampling rounds before ending at a one voxel cell.
            Default: 12
        Returns
        -------
        result: Dict[float, np.array]
            Dictionary of reduced arrays.
            Keys are reduction fold, values the reduced array.
    """

    # how much are we downsampling per round
    fold = gmean(block_size)

    # original image
    i = 0
    pyramid = {fold ** i: input_arr.copy()}

    # downsample and save to dict
    i = 1
    while (i <= max_iters) and (np.max(pyramid[fold ** (i - 1)].shape) > 1):
        pyramid[fold ** i] = block_reduce(pyramid[fold ** (i - 1)], block_size, func)
        i += 1

    return pyramid


def pyramid_correlation(img1, img2, **pyramid_kwargs):

    pyramid_1 = blockreduce_pyramid(img1, **pyramid_kwargs)
    pyramid_2 = blockreduce_pyramid(img2, **pyramid_kwargs)

    assert pyramid_1.keys() == pyramid_2.keys()

    corrs = {}
    for k in sorted(set({**pyramid_1, **pyramid_2}.keys())):
        assert pyramid_1[k].shape == pyramid_2[k].shape

        imgs_size_1 = (pyramid_1[k].size == 1) & (pyramid_2[k].size == 1)
        imgs_same = np.all(pyramid_1[k] == pyramid_2[k])
        stdv_1_zero = np.all(pyramid_1[k]) | np.all(~pyramid_1[k])
        stdv_2_zero = np.all(pyramid_2[k]) | np.all(~pyramid_2[k])

        if imgs_size_1 | imgs_same:
            corrs[k] = 1.0
        elif stdv_1_zero | stdv_2_zero:
            corrs[k] = 0.0
        else:
            corrs[k], _ = pearsonr(pyramid_1[k].flatten(), pyramid_2[k].flatten())

    return corrs


def draw_pairs(input_list, n_pairs=1):
    """
    Draw unique (ordered) pairs of examples from input_list at random.
    Input list is not a list of pairs, just a list of single exemplars.
    
    Example:
        >>> draw_pairs([0,1,2,3], n_pairs=3)
        >>> {(1,2), (2,3), (0,3)}
        
    Note:
        A pair is only unique up to order, e.g. (1,2) == (2,1).  this function
        only returns and compared sorted tuple to handle this 
    """
    
    # make sure requested number of uniquepairs in possible
    L = len(input_list)
    assert n_pairs <= L*(L-1)/2
    
    # draw n_pairs of size 2 sets from input_list    
    pairs = set()
    while len(pairs) < n_pairs:
        pairs |= {frozenset(sorted(np.random.choice(input_list, 2 ,replace=False)))}
    
    # return a set of ordered tuples to not weird people out
    return {tuple(sorted(p)) for p in pairs}