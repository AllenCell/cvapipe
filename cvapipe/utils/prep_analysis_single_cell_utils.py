#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import math
import time

from shutil import rmtree
import re
from scipy.ndimage import zoom
from scipy.ndimage.measurements import center_of_mass
from skimage.morphology import dilation, ball
from skimage.measure import regionprops, label
import aicsimageio
from aicsimageio import AICSImage, imread
from aicsimageio.writers import ome_tiff_writer as save_tif
from aicsimageprocessing import resize, resize_to


def get_largest_cc(labels: np.ndarray) -> np.ndarray:
    """
    find the largest connected component in a labeled image

    Parameters
    ----------
    label: np.ndarray
        a labeled image

    Returns
    -------
    largestCC: np.ndarray
        a binary image of the same size as input, only
        the largest connect component is kept in the image
    """
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def find_true_edge_cells(mem_seg_whole_valid: np.ndarray) -> List:
    """
    find the indices of true edge cells

    Parameters
    ----------
    mem_seg_whole_valid: np.ndarray
        a 3D labeled image of cell segmentation

    Returns
    -------
    edge_idx: List
        a list of indices of edge cells
    """

    mem_coverage = np.amax(mem_seg_whole_valid, axis=0)
    FOV_size = mem_coverage.shape[0] * mem_coverage.shape[1]
    mem_coverage_ratio = np.count_nonzero(mem_coverage > 0) / FOV_size
    if mem_coverage_ratio > 0.95:
        # not enough plate area, so no edge cell
        return []

    # extract body chunk
    colony_top = mem_seg_whole_valid.shape[0] - 1
    for zz in np.arange(
        mem_seg_whole_valid.shape[0] - 1, mem_seg_whole_valid.shape[0] // 2 + 1, -1
    ):
        if (
            np.count_nonzero(mem_seg_whole_valid[zz, :, :] > 0) / FOV_size
            > 0.5 * mem_coverage_ratio
        ):
            colony_top = zz
            break

    colony_bottom = 0
    for zz in np.arange(3, mem_seg_whole_valid.shape[0] // 2):
        if (
            np.count_nonzero(mem_seg_whole_valid[zz, :, :] > 0) / FOV_size
            > 0.5 * mem_coverage_ratio
        ):
            colony_bottom = zz
            break

    # make sure not super flat
    if colony_top - colony_bottom < 5:
        return []

    # find the list of true edge cells
    colony_middle = (colony_top - colony_bottom) // 2 + colony_bottom
    body_chunk = mem_seg_whole_valid[colony_middle - 2 : colony_middle + 3, :, :]

    # extract cover slip object
    bg_in_body_chunk = body_chunk == 0
    bg_label_in_body_chunk = label(bg_in_body_chunk)
    outer_bg = get_largest_cc(bg_label_in_body_chunk)

    # find cells touching cover slip
    outer_bg_cover = dilation(outer_bg, ball(5))
    edge_idx = list(np.unique(body_chunk[outer_bg_cover > 0]))

    return edge_idx


def build_one_cell_for_classification(crop_raw, mem_seg, down_ratio=0.5):
    """
    build a single file for one cell, to be used by mitotic classifier

    Parameters
    ----------
    crop_raw: numpy.array
        croped raw image (multi-channel with order: dna, mem, struct)
    mem_seg: numpy.array
        cell segmentation of the cell (same ZYX shape as crop_raw)
    down_ratio: float
        the downsampling ratio the mitotic classifier is trained on
        Defaul: 0.5

    Returns
    -------
    edge_idx: a list of indices of edge cells
    """

    # raw image normalization
    img_raw = crop_raw[0, :, 0:2, :, :].astype(np.float32)
    img_raw = np.transpose(img_raw, (1, 0, 2, 3))
    for dim in range(0, 2):
        img_ch = img_raw[dim, :, :, :].copy()
        low = np.percentile(img_ch, 0.05)
        upper = np.percentile(img_ch, 99.5)
        img_ch[img_ch > upper] = upper
        img_ch[img_ch < low] = low
        img_ch = (img_ch - low) / (upper - low)
        img_raw[dim, :, :, :] = img_ch

    # load seg
    mem_seg_tight = dilation(mem_seg > 0, ball(3))

    z_range = np.where(np.any(mem_seg_tight, axis=(1, 2)))
    y_range = np.where(np.any(mem_seg_tight, axis=(0, 2)))
    x_range = np.where(np.any(mem_seg_tight, axis=(0, 1)))
    z_range = z_range[0]
    y_range = y_range[0]
    x_range = x_range[0]

    roi = [
        max(z_range[0] - 2, 0),
        min(z_range[-1] + 4, mem_seg_tight.shape[0]),
        max(y_range[0] - 5, 0),
        min(y_range[-1] + 5, mem_seg_tight.shape[1]),
        max(x_range[0] - 5, 0),
        min(x_range[-1] + 5, mem_seg_tight.shape[2]),
    ]

    mem_seg_tight = mem_seg_tight[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]

    mem_img = img_raw[1, roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
    dna_img = img_raw[0, roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]

    mem_seg_tight = zoom(mem_seg_tight, down_ratio, order=0)
    mem_img = zoom(mem_img, down_ratio, order=2)
    dna_img = zoom(dna_img, down_ratio, order=2)

    mem_img[mem_seg_tight == 0] = 0
    dna_img[mem_seg_tight == 0] = 0

    # merge seg and raw
    img_out = np.stack((dna_img, mem_img), axis=0)

    return img_out


def euc_dist_3d(p1, p2):
    dist = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5
    return dist


def euc_dist_2d(p1, p2):
    dist = ((p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5
    return dist


def overlap_area(p1, p2):
    matrix = np.zeros((3, 7, 7))
    matrix[1, :, :] = 1
    d1 = dilation(p1, selem=matrix)
    d2 = dilation(p2, selem=matrix)
    area = np.count_nonzero(np.logical_and(d1, d2))
    return area


def single_cell_gen_one_fov(
    row_index: int, row: pd.Series, single_cell_dir: Path, overwrite: bool = False
) -> List:
    # TODO: currently, overwrite flag is not working.
    # need to think more on how to deal with overwrite
    ########################################
    # parameters
    ########################################
    # Don't use dask for image reading
    aicsimageio.use_dask(False)
    standard_res_qcb = 0.108

    print(f"ready to process FOV: {row.FOVId}")

    ########################################
    # load image and segmentation
    ########################################
    if row.AlignedImageReadPath is None:
        raw_fn = row.SourceReadPath
    else:
        raw_fn = row.AlignedImageReadPath

    # verify filepaths
    if not (
        os.path.exists(raw_fn)
        and os.path.exists(row.MembraneSegmentationReadPath)
        and os.path.exists(row.StructureSegmentationReadPath)
    ):
        # fail
        return [row.FOVId, True, "missing segmentation or raw files"]

    # get the raw image and split into different channels
    start_time = time.time()
    raw_data = np.squeeze(AICSImage(raw_fn).data)
    raw_mem0 = raw_data[int(row.ChannelNumber638), :, :, :]
    raw_nuc0 = raw_data[int(row.ChannelNumber405), :, :, :]
    # find valid structure channel index
    if math.isnan(row.ChannelNumber561):
        raw_struct0 = raw_data[int(row.ChannelNumber488), :, :, :]
    else:
        raw_struct0 = raw_data[int(row.ChannelNumber561), :, :, :]
    total_t = time.time() - start_time
    print(f"Raw image load in: {total_t} sec")

    """
    # because the seg results are save in one file, MembraneSegmentationFilename
    # and NucleusSegmentationFilename should be the same

    # temporily comment out during test with old data.
    # new data will need this assertion
    try:
        assert row.MembraneSegmentationFilename == row.NucleusSegmentationFilename,\
            f"MembraneSegmentationFilename: {row.MembraneSegmentationFilename} and \
            NucleusSegmentationFilename: {row.NucleusSegmentationFilename} mismatch"
    except AssertionError as e:
        log.info(
            f"Failed single cell generation for FOVId: {row.FOVId}. Error: {e}"
        )
        return SingleCellGenOneFOVFailure(row.FOVId, True, str(e))

    # this will be used for new data
    seg_reader = AICSImage(row.MembraneSegmentationReadPath)
    nuc_seg_whole = seg_reader.get_image_data("ZYX", S=0, T=0, C=0) 
    mem_seg_whole = seg_reader.get_image_data("ZYX", S=0, T=0, C=1) 
    """

    # HACK: temporary solution when testing with old data ###
    nuc_seg_whole = np.squeeze(imread(row.NucleusSegmentationReadPath))
    mem_seg_whole = np.squeeze(imread(row.MembraneSegmentationReadPath))

    # get structure segmentation
    struct_seg_whole = np.squeeze(imread(row.StructureSegmentationReadPath))

    print(f"Segmentation load successfully: {row.FOVId}")

    #########################################
    # run single cell qc in this fov
    #########################################
    # try:
    ######################
    min_mem_size = 70000
    min_nuc_size = 10000
    ######################

    # flag for any segmented object in this FOV removed as bad cells
    full_fov_pass = 1

    # double check big failure, quick reject
    if mem_seg_whole.max() <= 3 or nuc_seg_whole.max() <= 3:
        # bad images, but not bug, use "False"
        return [row.FOVId, False, "very few cells segmented"]

    # prune the results (remove cells touching image boundary)
    boundary_mask = np.zeros_like(mem_seg_whole)
    boundary_mask[:, :3, :] = 1
    boundary_mask[:, -3:, :] = 1
    boundary_mask[:, :, :3] = 1
    boundary_mask[:, :, -3:] = 1
    bd_idx = list(np.unique(mem_seg_whole[boundary_mask > 0]))

    # maintain a valid cell list, initialize with all cells minus
    # cells touching the image boundary, and minus cells with
    # no record in labkey (e.g., manually removed based on user's feedback)
    all_cell_index_list = list(np.unique(mem_seg_whole[mem_seg_whole > 0]))
    full_set_with_no_boundary = set(all_cell_index_list) - set(bd_idx)
    set_not_in_labkey = full_set_with_no_boundary - set(row.index_to_id_dict[0].keys())
    valid_cell = list(full_set_with_no_boundary - set_not_in_labkey)

    # single cell QC
    valid_cell_0 = valid_cell.copy()
    for list_idx, this_cell_index in enumerate(valid_cell):
        single_mem = mem_seg_whole == this_cell_index
        single_nuc = nuc_seg_whole == this_cell_index

        # remove too small cells from valid cell list
        if (
            np.count_nonzero(single_mem) < min_mem_size
            or np.count_nonzero(single_nuc) < min_nuc_size
        ):
            valid_cell_0.remove(this_cell_index)
            full_fov_pass = 0

            # no need to go to next QC criteria
            continue

        # make sure the cell is not leaking to the bottom or top
        z_range_single = np.where(np.any(single_mem, axis=(1, 2)))
        single_min_z = z_range_single[0][0]
        single_max_z = z_range_single[0][-1]

        if single_min_z == 0 or single_max_z >= single_mem.shape[0] - 1:
            valid_cell_0.remove(this_cell_index)
            full_fov_pass = 0
    valid_cell = valid_cell_0.copy()

    # if only one cell left or no cell left, just throw it away
    # HACK: use 0 during testing, change to 1 in real run
    if len(valid_cell_0) < 1:
        return [row.FOVId, False, "very few cells left after single cell QC"]

    print(f"single cell QC done in FOV: {row.FOVId}")

    #################################################################
    # resize the image into isotropic dimension
    #################################################################
    raw_nuc = resize(
        raw_nuc0,
        (
            row.PixelScaleZ / standard_res_qcb,
            row.PixelScaleY / standard_res_qcb,
            row.PixelScaleX / standard_res_qcb,
        ),
        method="bilinear",
    ).astype(np.uint16)

    raw_mem = resize(
        raw_mem0,
        (
            row.PixelScaleZ / standard_res_qcb,
            row.PixelScaleY / standard_res_qcb,
            row.PixelScaleX / standard_res_qcb,
        ),
        method="bilinear",
    ).astype(np.uint16)

    raw_str = resize(
        raw_struct0,
        (
            row.PixelScaleZ / standard_res_qcb,
            row.PixelScaleY / standard_res_qcb,
            row.PixelScaleX / standard_res_qcb,
        ),
        method="bilinear",
    ).astype(np.uint16)

    mem_seg_whole = resize_to(mem_seg_whole, raw_mem.shape, method="nearest")
    nuc_seg_whole = resize_to(nuc_seg_whole, raw_nuc.shape, method="nearest")
    struct_seg_whole = resize_to(struct_seg_whole, raw_str.shape, method="nearest")

    #################################################################
    # calculate fov related info
    #################################################################
    index_to_cellid_map = dict()
    cellid_to_index_map = dict()
    for list_idx, this_cell_index in enumerate(valid_cell):
        # this is always valid since indices not in index_to_id_dict.keys()
        # have been removed
        index_to_cellid_map[this_cell_index] = row.index_to_id_dict[0][this_cell_index]
    for index_dict, cellid_dict in index_to_cellid_map.items():
        cellid_to_index_map[cellid_dict] = index_dict

    # compute center of mass
    index_to_centroid_map = dict()
    center_list = center_of_mass(nuc_seg_whole > 0, nuc_seg_whole, valid_cell)
    for list_idx, this_cell_index in enumerate(valid_cell):
        index_to_centroid_map[this_cell_index] = center_list[list_idx]

    # compute whole stack min/max z
    mem_seg_whole_valid = np.zeros_like(mem_seg_whole)
    for list_idx, this_cell_index in enumerate(valid_cell):
        mem_seg_whole_valid[mem_seg_whole == this_cell_index] = this_cell_index
    z_range_whole = np.where(np.any(mem_seg_whole_valid, axis=(1, 2)))
    stack_min_z = z_range_whole[0][0]
    stack_max_z = z_range_whole[0][-1]

    # find true edge cells, the cells in the outer layer of a colony
    true_edge_cells = []
    edge_fov_flag = False
    if row.ColonyPosition is None:
        # parse colony position from file name
        reg = re.compile("(-|_)((\d)?)(e)((\d)?)(-|_)")  # noqa: W605
        if reg.search(os.path.basename(raw_fn)):
            edge_fov_flag = True
    else:
        if row.ColonyPosition.lower() == "edge":
            edge_fov_flag = True

    if edge_fov_flag:
        true_edge_cells = find_true_edge_cells(mem_seg_whole_valid)

    #################################################################
    # calculate a dictionary to store FOV info
    #################################################################
    df_fov_meta = {
        "FOVId": row.FOVId,
        "structure_name": row.Gene,
        "position": row.ColonyPosition,
        "raw_fn": raw_fn,
        "str_filename": row.StructureSegmentationReadPath,
        "mem_seg_fn": row.MembraneSegmentationReadPath,
        "nuc_seg_fn": row.NucleusSegmentationReadPath,
        "index_to_id_dict": [index_to_cellid_map],
        "id_to_index_dict": [cellid_to_index_map],
        "xy_res": row.PixelScaleX,
        "z_res": row.PixelScaleZ,
        "stack_min_z": stack_min_z,
        "stack_max_z": stack_max_z,
        "scope_id": row.InstrumentId,
        "well_id": row.WellId,
        "well_name": row.WellName,
        "plateId": row.PlateId,
        "passage": row.Passage,
        "image_size": [list(raw_mem.shape)],
        "fov_seg_pass": full_fov_pass,
    }

    print(f"FOV info is done: {row.FOVId}, ready to loop through cells")

    # loop through all valid cells in this fov
    df_cell_meta = []
    for list_idx, this_cell_index in enumerate(valid_cell):
        nuc_seg = nuc_seg_whole == this_cell_index
        mem_seg = mem_seg_whole == this_cell_index

        ###########################
        # implement nbr info
        ###########################
        single_mem_dilate = dilation(mem_seg, selem=ball(3))
        whole_template = mem_seg_whole.copy()
        whole_template[mem_seg] = 0
        this_cell_nbr_candiate_list = list(
            np.unique(whole_template[single_mem_dilate > 0])
        )
        this_cell_nbr_dist_3d = []
        this_cell_nbr_dist_2d = []
        this_cell_nbr_overlap_area = []
        this_cell_nbr_complete = 1

        for nbr_index, nbr_id in enumerate(this_cell_nbr_candiate_list):
            if nbr_id == 0 or nbr_id == this_cell_index:
                continue
            elif not (nbr_id in valid_cell):
                this_cell_nbr_complete = 0
                continue

            # only do calculation for valid neighbors
            nuc_dist_3d = euc_dist_3d(
                index_to_centroid_map[nbr_id], index_to_centroid_map[this_cell_index]
            )
            nuc_dist_2d = euc_dist_2d(
                index_to_centroid_map[nbr_id], index_to_centroid_map[this_cell_index]
            )
            overlap = overlap_area(mem_seg, mem_seg_whole == nbr_id)
            this_cell_nbr_dist_3d.append((index_to_cellid_map[nbr_id], nuc_dist_3d))
            this_cell_nbr_dist_2d.append((index_to_cellid_map[nbr_id], nuc_dist_2d))
            this_cell_nbr_overlap_area.append((index_to_cellid_map[nbr_id], overlap))
        if len(this_cell_nbr_dist_3d) == 0:
            this_cell_nbr_complete = 0

        # get cell id
        cell_id = index_to_cellid_map[this_cell_index]

        # make the path for saving single cell crop result
        thiscell_path = single_cell_dir / Path(str(cell_id))
        if os.path.isdir(thiscell_path):
            rmtree(thiscell_path)
        os.mkdir(thiscell_path)

        ###############################
        # compute and  generate crop
        ###############################
        # determine crop roi
        z_range = np.where(np.any(mem_seg, axis=(1, 2)))
        y_range = np.where(np.any(mem_seg, axis=(0, 2)))
        x_range = np.where(np.any(mem_seg, axis=(0, 1)))
        z_range = z_range[0]
        y_range = y_range[0]
        x_range = x_range[0]

        # define a large ROI based on bounding box
        roi = [
            max(z_range[0] - 10, 0),
            min(z_range[-1] + 12, mem_seg.shape[0]),
            max(y_range[0] - 40, 0),
            min(y_range[-1] + 40, mem_seg.shape[1]),
            max(x_range[0] - 40, 0),
            min(x_range[-1] + 40, mem_seg.shape[2]),
        ]

        # roof augmentation
        mem_nearly_top_z = int(
            z_range[0] + round(0.75 * (z_range[-1] - z_range[0] + 1))
        )
        mem_top_mask = np.zeros(mem_seg.shape, dtype=np.byte)
        mem_top_mask[mem_nearly_top_z:, :, :] = mem_seg[mem_nearly_top_z:, :, :] > 0
        mem_top_mask_dilate = dilation(
            mem_top_mask > 0, selem=np.ones((21, 1, 1), dtype=np.byte)
        )
        mem_top_mask_dilate[:mem_nearly_top_z, :, :] = (
            mem_seg[:mem_nearly_top_z, :, :] > 0
        )

        # crop mem/nuc seg
        mem_seg = mem_seg.astype(np.uint8)
        mem_seg = mem_seg[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
        mem_seg[mem_seg > 0] = 255

        nuc_seg = nuc_seg.astype(np.uint8)
        nuc_seg = nuc_seg[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
        nuc_seg[nuc_seg > 0] = 255

        mem_top_mask_dilate = mem_top_mask_dilate.astype(np.uint8)
        mem_top_mask_dilate = mem_top_mask_dilate[
            roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]
        ]
        mem_top_mask_dilate[mem_top_mask_dilate > 0] = 255

        # crop str seg (without roof augmentation)
        str_seg_crop = struct_seg_whole[
            roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]
        ].astype(np.uint8)
        str_seg_crop[mem_seg < 1] = 0
        str_seg_crop[str_seg_crop > 0] = 255

        # crop str seg (with roof augmentation)
        str_seg_crop_roof = struct_seg_whole[
            roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]
        ].astype(np.uint8)
        str_seg_crop_roof[mem_top_mask_dilate < 1] = 0
        str_seg_crop_roof[str_seg_crop_roof > 0] = 255

        # merge and save the cropped segmentation
        all_seg = np.stack(
            [nuc_seg, mem_seg, mem_top_mask_dilate, str_seg_crop, str_seg_crop_roof],
            axis=0,
        )
        all_seg = np.expand_dims(np.transpose(all_seg, (1, 0, 2, 3)), axis=0)

        crop_seg_path = thiscell_path / "segmentation.ome.tif"
        writer = save_tif.OmeTiffWriter(crop_seg_path)
        writer.save(all_seg)

        # crop raw image
        raw_nuc_thiscell = raw_nuc[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
        raw_mem_thiscell = raw_mem[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
        raw_str_thiscell = raw_str[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
        crop_raw_merged = np.expand_dims(
            np.stack((raw_nuc_thiscell, raw_mem_thiscell, raw_str_thiscell), axis=1),
            axis=0,
        )

        crop_raw_path = thiscell_path / "raw.ome.tif"
        writer = save_tif.OmeTiffWriter(crop_raw_path)
        writer.save(crop_raw_merged)

        ############################
        # check for pair
        ############################
        dist_cutoff = 85
        dna_label, dna_num = label(nuc_seg > 0, return_num=True)

        if dna_num < 2:
            # certainly not pair if there is only one cc
            this_cell_is_pair = 0
        else:
            stats = regionprops(dna_label)
            region_size = [stats[i]["area"] for i in range(dna_num)]
            large_two = sorted(
                range(len(region_size)), key=lambda sub: region_size[sub]
            )[-2:]
            dis = euc_dist_3d(
                stats[large_two[0]]["centroid"], stats[large_two[1]]["centroid"]
            )
            if dis > dist_cutoff:
                sz1 = stats[large_two[0]]["area"]
                sz2 = stats[large_two[1]]["area"]
                if sz1 / sz2 > 1.5625 or sz1 / sz2 < 0.64:
                    # the two parts do not have comparable sizes
                    this_cell_is_pair = 0
                else:
                    this_cell_is_pair = 1
            else:
                # not far apart enough
                this_cell_is_pair = 0

        name_dict = {
            "crop_raw": ["dna", "membrane", "structure"],
            "crop_seg": [
                "dna_segmentation",
                "membrane_segmentation",
                "membrane_segmentation",
                "struct_segmentation",
                "struct_segmentation",
            ],
        }

        # out for mitotic classifier
        img_out = build_one_cell_for_classification(crop_raw_merged, mem_seg)
        out_fn = thiscell_path / "for_mito_prediction.npy"
        np.save(out_fn, img_out)

        #########################################
        if len(true_edge_cells) > 0 and (this_cell_index in true_edge_cells):
            this_is_edge_cell = 1
        else:
            this_is_edge_cell = 0

        # write qcb cell meta
        df_cell_meta.append(
            {
                "CellId": cell_id,
                "structure_name": row.Gene,
                "pair": this_cell_is_pair,
                "this_cell_nbr_complete": this_cell_nbr_complete,
                "this_cell_nbr_dist_3d": [this_cell_nbr_dist_3d],
                "this_cell_nbr_dist_2d": [this_cell_nbr_dist_2d],
                "this_cell_nbr_overlap_area": [this_cell_nbr_overlap_area],
                "roi": [roi],
                "crop_raw": crop_raw_path,
                "crop_seg": crop_seg_path,
                "name_dict": [name_dict],
                "scale_micron": [[0.108, 0.108, 0.108]],
                "edge_flag": this_is_edge_cell,
                "fov_id": row.FOVId,
                "fov_path": raw_fn,
                "stack_min_z": stack_min_z,
                "stack_max_z": stack_max_z,
                "image_size": [list(raw_mem.shape)],
                "plateId": row.PlateId,
                "position": row.ColonyPosition,
                "scope_id": row.InstrumentId,
                "well_id": row.WellId,
                "well_name": row.WellName,
                "passage": row.Passage,
            }
        )
        print(f"Cell {cell_id} is done")

    #  single cell generation succeeds in this FOV
    print(f"FOV {row.FOVId} is done")
    return [df_fov_meta, df_cell_meta]
