from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, gmean
from skimage.measure import block_reduce
from skimage.util import crop

import matplotlib.pyplot as plt
import seaborn as sns

from aicsimageio import AICSImage


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


def safe_pearsonr(arr1, arr2):
    """Sensibly handle degenerate cases."""
    assert arr1.shape == arr2.shape

    imgs_same = np.all(arr1 == arr2)
    stdv_1_zero = len(np.unique(arr1)) == 1
    stdv_2_zero = len(np.unique(arr2)) == 1

    if (stdv_1_zero | stdv_2_zero) & imgs_same:
        corr = 1.0
    elif (stdv_1_zero | stdv_2_zero) & (not imgs_same):
        corr = 0.0
    else:
        corr, _ = pearsonr(arr1, arr2)

    return corr


def pyramid_correlation(img1, img2, mask=None, permute=False, **pyramid_kwargs):

    # make sure inputs are all the same shape
    assert img1.shape == img2.shape
    if mask is None:
        mask = np.ones_like(img1)
    assert mask.shape == img1.shape

    # make image pyramids
    pyramid_1 = blockreduce_pyramid(img1, **pyramid_kwargs)
    pyramid_2 = blockreduce_pyramid(img2, **pyramid_kwargs)

    # also make a mask pyramid
    mask_kwargs = pyramid_kwargs.copy()
    mask_kwargs["func"] = np.max
    pyramid_mask = blockreduce_pyramid(mask, **mask_kwargs)

    # make sure everything has the same keys
    assert pyramid_1.keys() == pyramid_2.keys()
    assert pyramid_mask.keys() == pyramid_1.keys()

    # select just voxels in gfp where mask is True and flatten them
    pyramid_1_masked_flat = {
        k: v[pyramid_mask[k] > 0].flatten() for k, v in pyramid_1.items()
    }
    pyramid_2_masked_flat = {
        k: v[pyramid_mask[k] > 0].flatten() for k, v in pyramid_2.items()
    }

    # at each resolution, find corr
    if not permute:
        corrs = {
            k: safe_pearsonr(pyramid_1_masked_flat[k], pyramid_2_masked_flat[k])
            for k in sorted(
                set({**pyramid_1_masked_flat, **pyramid_2_masked_flat}.keys())
            )
        }
    else:
        # shuffle voxels in one pyramid if we want the permuted baseline
        pyramid_1_masked_flat_permuted = pyramid_1_masked_flat.copy()
        for k in pyramid_1_masked_flat_permuted.keys():
            np.random.shuffle(pyramid_1_masked_flat[k])
        corrs = {
            k: safe_pearsonr(
                pyramid_1_masked_flat_permuted[k], pyramid_2_masked_flat[k]
            )
            for k in sorted(
                set({**pyramid_1_masked_flat_permuted, **pyramid_2_masked_flat}.keys())
            )
        }

    return corrs


def get_cell_mask(image_path, crop_size=(64, 160, 96), cell_mask_channel_ind=1):
    """
    Take a path to a tiff and return the masked gfp 3d volume
    """

    # load image
    image = AICSImage(image_path)
    data_6d = image.data
    mask_3d = data_6d[0, 0, cell_mask_channel_ind, :, :, :]

    # crop to desired shape
    z_dim, y_dim, x_dim = mask_3d.shape
    z_desired, y_desired, x_desired = crop_size
    z_crop = (z_dim - z_desired) // 2
    y_crop = (y_dim - y_desired) // 2
    x_crop = (x_dim - x_desired) // 2
    mask_3d = crop(mask_3d, ((z_crop, z_crop), (y_crop, y_crop), (x_crop, x_crop)))
    assert mask_3d.shape == crop_size

    return mask_3d


def get_gfp_single_channel_img(image_path, crop_size=(64, 160, 96), gfp_channel_ind=0):
    """
    Take a path to a tiff and return the masked gfp 3d volume
    """

    # load image
    image = AICSImage(image_path)
    data_6d = image.data
    gfp_3d = data_6d[0, 0, gfp_channel_ind, :, :, :]

    # crop to desired shape
    z_dim, y_dim, x_dim = gfp_3d.shape
    z_desired, y_desired, x_desired = crop_size
    z_crop = (z_dim - z_desired) // 2
    y_crop = (y_dim - y_desired) // 2
    x_crop = (x_dim - x_desired) // 2
    gfp_3d = crop(gfp_3d, ((z_crop, z_crop), (y_crop, y_crop), (x_crop, x_crop)))
    assert gfp_3d.shape == crop_size

    return gfp_3d


def compute_distance_metric(
    row, mdata_cols, px_size=0.29, crop_size=(64, 160, 96),
):
    """
    Main function to loop over in distributed
    """

    # get data for cells i and j
    gfp_i = get_gfp_single_channel_img(row.GeneratedStructuePath_i, crop_size=crop_size)
    gfp_j = get_gfp_single_channel_img(row.GeneratedStructuePath_j, crop_size=crop_size)

    # get the mask for the cell
    mask = get_cell_mask(
        Path(row.save_dir) / Path(row.save_reg_path).name, crop_size=crop_size
    )

    # kill gfp intensity outside of mask
    gfp_i[mask == 0] = 0
    gfp_j[mask == 0] = 0

    # multi-res comparison
    pyr_corrs = pyramid_correlation(gfp_i, gfp_j, mask=mask, func=np.mean)

    # comparison when one input is permuted (as baseline correlation)
    pyr_corrs_permuted = pyramid_correlation(
        gfp_i, gfp_j, mask=mask, permute=True, func=np.mean
    )

    # grab correlations at each res in a df
    df_tmp_corrs = pd.DataFrame()
    for k, v in sorted(pyr_corrs.items()):
        tmp_stat_dict = {
            "Resolution (micrometers)": px_size * k,
            "Pearson Correlation": v,
            "Pearson Correlation permuted": pyr_corrs_permuted[k],
        }
        df_tmp_corrs = df_tmp_corrs.append(tmp_stat_dict, ignore_index=True)

    # label stats with cell ids
    df_tmp_corrs["CellId"] = row.CellId

    # and append row metadata
    df_row_tmp = row.to_frame().T
    df_row_tmp = df_row_tmp.merge(df_tmp_corrs)

    return df_row_tmp


def clean_up_results(dist_metric_results):
    """
    Clean up distributed results.
    """
    df_final = pd.DataFrame()
    for dataframes in dist_metric_results:
        for corr_dataframe in dataframes:
            if corr_dataframe is not None:
                df_final = df_final.append(corr_dataframe)

    # fix up final pairwise dataframe
    df_final = df_final.reset_index(drop=True)
    df_final["Pearson Correlation gain over random"] = (
        df_final["Pearson Correlation"] - df_final["Pearson Correlation permuted"]
    )

    return df_final


def make_plot(data, save_loc):
    """
    Seaborn plot of mean corr gain over random vs resolution.
    """
    sns.set(style="ticks", rc={"lines.linewidth": 1.0})
    fig = plt.figure(figsize=(10, 7))

    ax = sns.pointplot(
        x="Resolution (micrometers)",
        y="Pearson Correlation gain over random",
        hue="GeneratedStructureName_i",
        data=data,
        ci=95,
        capsize=0.2,
        palette="Set2",
    )
    ax.legend(
        loc="upper left", bbox_to_anchor=(0.05, 0.95), ncol=1, frameon=False,
    )
    sns.despine(
        offset=0, trim=True,
    )

    # save the plot
    fig.savefig(
        save_loc, format="png", dpi=300, transparent=True,
    )
