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


def pyramid_correlation(img1, img2, **pyramid_kwargs):
    
    pyramid_1 = blockreduce_pyramid(img1, **pyramid_kwargs)
    pyramid_2 = blockreduce_pyramid(img2, **pyramid_kwargs)
    
    assert pyramid_1.keys() == pyramid_2.keys()
    
    corrs = {}
    for k in sorted(set({**pyramid_1, **pyramid_2}.keys())):
        assert pyramid_1[k].shape == pyramid_2[k].shape
        
        imgs_size_1 = (pyramid_1[k].size == 1) & (pyramid_2[k].size == 1)
        imgs_same = np.all(pyramid_1[k] == pyramid_2[k])
        stdv_1_zero = len(np.unique(pyramid_1[k])) == 1
        stdv_2_zero = len(np.unique(pyramid_2[k])) == 1
        
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
    assert n_pairs <= L * (L - 1) / 2

    # draw n_pairs of size 2 sets from input_list
    pairs = set()
    while len(pairs) < n_pairs:
        pairs |= {frozenset(sorted(np.random.choice(input_list, 2, replace=False)))}

    # return a set of ordered tuples to not weird people out
    return {tuple(sorted(p)) for p in pairs}


def make_masked_gfp_image(
    image_path, crop_size=(64, 160, 96), gfp_channel_ind=4, cell_mask_channel_ind=1
):
    """
    Take a path to a tiff and return the masked gfp 3d volume
    """

    # load image
    image = AICSImage(image_path)
    data_6d = image.data

    # mask gfp by cell boundary
    gfp_3d = data_6d[0, 0, 4, :, :, :]
    mask_3d = data_6d[0, 0, 1, :, :, :]
    gfp_3d[mask_3d == 0] = 0

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
    row,
    mdata_cols,
    px_size=0.29,
    crop_size=(64, 160, 96),
    par_dir=Path("/allen/aics/modeling/ritvik/projects/actk/"),
):
    """
    Main function to loop over in distributed
    """

    # get data for cells i and j
    gfp_i = make_masked_gfp_image(par_dir / row.CellImage3DPath_i, crop_size=crop_size)
    gfp_j = make_masked_gfp_image(par_dir / row.CellImage3DPath_j, crop_size=crop_size)

    # multi-res comparison
    pyr_corrs = pyramid_correlation(gfp_i, gfp_j, func=np.mean)

    # comparison when one input is permuted (as baseline correlation)
    gfp_i_shuf = gfp_i.copy().flatten()
    np.random.shuffle(gfp_i_shuf)
    gfp_i_shuf = gfp_i_shuf.reshape(gfp_i.shape)
    pyr_corrs_permuted = pyramid_correlation(gfp_i_shuf, gfp_j, func=np.mean)

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
    df_tmp_corrs["CellId_i"] = row.CellId_i
    df_tmp_corrs["CellId_j"] = row.CellId_j

    # and append row metadata
    df_row_tmp = row.to_frame().T
    df_row_tmp = df_row_tmp.merge(df_tmp_corrs)

    return df_row_tmp


def make_pairs_df(
    df_in,
    structs=[
        "Actin filaments",
        "Mitochondria",
        "Microtubules",
        "Nuclear envelope",
        "Desmosomes",
        "Plasma membrane",
        "Nucleolus (Granular Component)",
        "Nuclear pores",
    ],
    N_pairs_per_struct=100,
    mdata_cols=[
        "StructureShortName",
        "FOVId",
        "CellIndex",
        "CellId",
        "StandardizedFOVPath",
        "CellImage3DPath",
    ],
):
    """
    Make the pairwise df to loop over.
    """
    df_out = pd.DataFrame()
    for struct in structs:
        try:
            df_struct = df_in[df_in.StructureShortName == struct]
            pair_inds = draw_pairs(df_struct.index, n_pairs=N_pairs_per_struct)
            for (i, j) in pair_inds:
                row_i = df_struct.loc[i]
                row_j = df_struct.loc[j]
                row = (
                    row_i[mdata_cols]
                    .add_suffix("_i")
                    .append(row_j[mdata_cols].add_suffix("_j"))
                )
                df_out = df_out.append(row, ignore_index=True)
        except Exception as e:
            print(
                f"Not enough {struct} rows to get {N_pairs_per_struct} samples"
                f" Error {e}."
            )
            break
    assert np.all(df_out["CellId_i"] != df_out["CellId_j"])
    df_out = df_out.reset_index(drop=True)
    return df_out


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
    df_final = df_final.rename(
        columns={"StructureShortName_i": "StructureShortName"}
    ).drop(columns="StructureShortName_j")
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
        hue="StructureShortName",
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
