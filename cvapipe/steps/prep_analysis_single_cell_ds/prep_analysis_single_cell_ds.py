#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, NamedTuple

import dask.dataframe as dd
import pandas as pd
import numpy as np
import itertools
import math
import re
import time
from shutil import rmtree
import pyarrow.parquet as pq
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import regionprops, label
from skimage.morphology import dilation, ball

import aicsimageio
from aicsimageio import AICSImage, imread
from aicsimageio.writers import ome_tiff_writer as save_tif
from aicsimageprocessing import resize, resize_to
from aics_dask_utils import DistributedHandler
from datastep import Step, log_run_params

from ...constants import DatasetFields
from cvapipe.utils.prep_analysis_single_cell_utils import \
    find_true_edge_cells, build_one_cell_for_classification,\
    euc_dist_3d, euc_dist_2d, overlap_area

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class SingleCellGenOneFOVResult(NamedTuple):
    fov_id: Union[int, str]
    fov_meta: Dict
    cell_meta: List[Dict]


class SingleCellGenOneFOVFailure(NamedTuple):
    fov_id: int
    # flag for quit due to bug
    # True = likely a bug, False = failed QC (everything else is okay)
    # because sometimes the quit may be due to failed QC, which is acceptable
    # we want to make sure we catch all quit due to real issues
    bug_fail: bool 
    error: str


class PrepAnalysisSingleCellDs(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @staticmethod
    def _load_image_and_seg(
        row: pd.Series
    ) -> List:
        """
        load images and segmentations

        Parameters:
        ----------------------------
        row: pd.Series
            the row being processed from the whole fov dataset

        Return:
        -----------------------------
        raw_fn: Path
            path to raw image
        raw_mem0: np.ndarray
            raw membrane image
        raw_nuc0: np.ndarray
            raw nucleue image
        raw_struct0: np.ndarray
            raw structure image
        mem_seg_whole: np.ndarray
            cell segmentation image
        nuc_seg_whole: np.ndarray
            nucleus segmentation image
        struct_seg_whole: np.ndarray
            structure segmentation image
        """

        # select proper raw file to use
        # (pipeline 4.4 needs to use aligned images, but pipline 4.0-4.3 does not)
        if row.AlignedImageReadPath is None:
            raw_fn = row.SourceReadPath
        else:
            raw_fn = row.AlignedImageReadPath

        # verify filepaths 
        assert os.path.exists(raw_fn), f"original image not found: {raw_fn}"
        assert os.path.exists(row.MembraneSegmentationReadPath),\
            f"cell segmentation not found: {row.MembraneSegmentationReadPath}"
        assert os.path.exists(row.StructureSegmentationReadPath),\
            f"structure segmentation not found {row.StructureSegmentationReadPath}"        

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
        log.info(f"Raw image load in: {total_t} sec")

        '''
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
        '''

        # HACK: temporary solution when testing with old data ###
        nuc_seg_whole = np.squeeze(imread(row.NucleusSegmentationReadPath))
        mem_seg_whole = np.squeeze(imread(row.MembraneSegmentationReadPath))

        # get structure segmentation
        struct_seg_whole = np.squeeze(imread(row.StructureSegmentationReadPath))

        return [raw_fn, raw_mem0, raw_nuc0, raw_struct0, mem_seg_whole, 
                nuc_seg_whole, struct_seg_whole]

    @staticmethod
    def _single_cell_qc_in_one_fov(
        mem_seg_whole: np.ndarray,
        nuc_seg_whole: np.ndarray,
        row: pd.Series
    ) -> List:
        """
        make sure the cell/nucleus segmentation not failed terribly
        and remove the bad cells when possible (e.g., very small cells)

        Parameters:
        ----------------------------
        mem_seg_whole: np.ndarray
            labeled image of cell segmentation
        nuc_seg_whole: np.ndarray
            labeled image of nucleus segmentation
        row: pd.Series
            the row being processed from the whole fov dataset

        Return:
        -----------------------------
        full_fov_pass: int
            flag for whether all cells in this fov can be trusted
        valid_cell: List
            the list of CellIndex after removing bad cells
        """

        ################################################################
        min_mem_size = 70000
        min_nuc_size = 10000
        ################################################################

        # flag for any segmented object in this FOV removed as bad cells
        full_fov_pass = 1

        # double check big failure, quick reject
        assert mem_seg_whole.max() > 3 and nuc_seg_whole.max() > 3,\
            f"very few cells segmented in {row.MembraneSegmentationReadPath}"

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
        set_not_in_labkey = full_set_with_no_boundary - \
            set(row.index_to_id_dict[0].keys())
        valid_cell = list(full_set_with_no_boundary - set_not_in_labkey)

        # single cell QC
        valid_cell_0 = valid_cell.copy() 
        for list_idx, this_cell_index in enumerate(valid_cell):
            single_mem = mem_seg_whole == this_cell_index
            single_nuc = nuc_seg_whole == this_cell_index

            # remove too small cells from valid cell list
            if np.count_nonzero(single_mem) < min_mem_size or \
               np.count_nonzero(single_nuc) < min_nuc_size:
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

        # if only one cell left or no cell left, just throw it away
        # HACK: use 0 during testing, change to 1 in real run
        assert len(valid_cell_0) > 0, \
            f"very few cells left after single cell QC in {row.FOVId}"

        return [full_fov_pass, valid_cell]

    @staticmethod
    def _calculate_fov_info(
        row: pd.Series,
        valid_cell: List,
        nuc_seg_whole: np.ndarray,
        mem_seg_whole: np.ndarray,
        raw_fn: Path
    ) -> List:
        """

        calulate basic info related to the whole FOV

        Parameter:
        -----------------------
        row: pd.Series
            current row being processing from the fov dataset
        valid_cell: List
            the list of valid CellIndex
        nuc_seg_whole: np.ndarray
            labeled image of nucleus segmentation
        mem_seg_whole: np.ndarray
            labeled image of cell segmentation
        raw_fn: Path
            path to raw image

        Return:
        -----------------------
        index_to_cellid_map: Dict
            mapping from CellIndex to CellID
        cellid_to_index_map: Dcit
            mapping from CellId to CellIndex
        index_to_centroid_map: Dict
            mapping from CellIndex to its centroid position
        stack_min_z: int
            the minimum z of all cells in this FOV
        stack_max_z: int
             the maximum z of all cells in this FOV
        true_edge_cells: List
            the list of CellIndex that is truely on edge of the colony
        """

        # identify index and CellId 1:1 mapping
        index_to_cellid_map = dict()
        cellid_to_index_map = dict()
        for list_idx, this_cell_index in enumerate(valid_cell):
            # this is always valid since indices not in index_to_id_dict.keys() 
            # have been removed
            index_to_cellid_map[this_cell_index] = \
                row.index_to_id_dict[0][this_cell_index]
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
            reg = re.compile('(-|_)((\d)?)(e)((\d)?)(-|_)')
            if reg.search(os.path.basename(raw_fn)):
                edge_fov_flag = True
        else:
            if row.ColonyPosition.lower() == 'edge':
                edge_fov_flag = True

        if edge_fov_flag:
            true_edge_cells = find_true_edge_cells(mem_seg_whole_valid)

        return [index_to_cellid_map, cellid_to_index_map, index_to_centroid_map,
                stack_min_z, stack_max_z, true_edge_cells]

    @staticmethod
    def _get_roi_and_crop(
        mem_seg: np.ndarray,
        nuc_seg: np.ndarray,
        struct_seg_whole: np.ndarray
    ) -> List:
        """
        calculate roi based on cell segmentaiton, and get two
        versions of crop segmentation (one with roof augmentation,
        one without roof augmentation). 

        Roof augmentation is artifical dilation of cell segmentation 
        near top to make sure all structure near the top can be included

        Paramter: 
        ----------------------
        mem_seg: np.ndarray
            binary image of cell segmentation of this cell

        nuc_seg: np.ndarray
            binary image of nucleus segmentation of this cell

        struct_seg_whole: np.ndarray
            binary image of structure segmentation of this FOV

        Return:
        ----------------------
        roi: List
            roi to crop for this cell as a list of 
            [left_z, right_z, left_y, right_y, left_x, right_x]

        nuc_seg: np.ndarray,
            cropped segmentation of nucleus
        mem_seg: np.ndarray,
            cropped segmentation of cell (no roof augmentation)
        mem_top_mask_dilate: np.ndarray,
            cropped segmentation of cell (with roof augmentation)
        str_seg_crop: np.ndarray
            cropped segmentation of structure (no roof augmentatiuon)
        str_seg_crop_roof: np.ndarray
            cropped segmentation of structure (with roof augmentatiuon)
        """
        # determine crop roi
        z_range = np.where(np.any(mem_seg, axis=(1, 2)))
        y_range = np.where(np.any(mem_seg, axis=(0, 2)))
        x_range = np.where(np.any(mem_seg, axis=(0, 1)))
        z_range = z_range[0]
        y_range = y_range[0]
        x_range = x_range[0]

        # define a large ROI based on bounding box
        roi = [max(z_range[0] - 10, 0), min(z_range[-1] + 12, mem_seg.shape[0]),
               max(y_range[0] - 40, 0), min(y_range[-1] + 40, mem_seg.shape[1]),
               max(x_range[0] - 40, 0), min(x_range[-1] + 40, mem_seg.shape[2])]

        # roof augmentation
        mem_nearly_top_z = int(z_range[0] + round(0.75 * (
            z_range[-1] - z_range[0] + 1)))
        mem_top_mask = np.zeros(mem_seg.shape, dtype=np.byte)
        mem_top_mask[mem_nearly_top_z:, :, :] = \
            mem_seg[mem_nearly_top_z:, :, :] > 0                  
        mem_top_mask_dilate = dilation(mem_top_mask > 0,
                                       selem=np.ones((21, 1, 1), dtype=np.byte))
        mem_top_mask_dilate[:mem_nearly_top_z, :, :] = \
            mem_seg[: mem_nearly_top_z, :, :] > 0

        # crop mem/nuc seg
        mem_seg = mem_seg.astype(np.uint8)
        mem_seg = mem_seg[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
        mem_seg[mem_seg > 0] = 255

        nuc_seg = nuc_seg.astype(np.uint8)
        nuc_seg = nuc_seg[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
        nuc_seg[nuc_seg > 0] = 255

        mem_top_mask_dilate = mem_top_mask_dilate.astype(np.uint8)
        mem_top_mask_dilate = mem_top_mask_dilate[roi[0]:roi[1], roi[2]:roi[3], 
                                                  roi[4]:roi[5]]
        mem_top_mask_dilate[mem_top_mask_dilate > 0] = 255

        # crop str seg (without roof augmentation)
        str_seg_crop = struct_seg_whole[roi[0]:roi[1], roi[2]:roi[3], 
                                        roi[4]:roi[5]].astype(np.uint8)
        str_seg_crop[mem_seg < 1] = 0
        str_seg_crop[str_seg_crop > 0] = 255

        # crop str seg (with roof augmentation)
        str_seg_crop_roof = struct_seg_whole[roi[0]:roi[1], roi[2]:roi[3], 
                                             roi[4]:roi[5]].astype(np.uint8)
        str_seg_crop_roof[mem_top_mask_dilate < 1] = 0
        str_seg_crop_roof[str_seg_crop_roof > 0] = 255

        return [roi, nuc_seg, mem_seg, mem_top_mask_dilate, 
                str_seg_crop, str_seg_crop_roof]

    @staticmethod
    def _check_if_pair(nuc_seg: np.ndarray) -> int:
        """

        check if this cell is a pair after division or not,
        based on nucleus segmentation

        Parameter:
        -----------------------
        nuc_seg: np.ndarray
            cropped segmentation of nucleus

        Return:
        ------------------------
        this_cell_is_pair: int
            flag for this cell being a pair or not
        """
        dist_cutoff = 85
        dna_label, dna_num = label(nuc_seg > 0, return_num=True)

        if dna_num < 2:
            # certainly not pair if there is only one cc 
            this_cell_is_pair = 0
        else:
            stats = regionprops(dna_label)
            region_size = [stats[i]['area'] for i in range(dna_num)]
            large_two = sorted(range(len(region_size)), 
                               key=lambda sub: region_size[sub])[-2:]
            dis = euc_dist_3d(stats[large_two[0]]['centroid'], 
                              stats[large_two[1]]['centroid'])
            if dis > dist_cutoff:
                sz1 = stats[large_two[0]]['area']
                sz2 = stats[large_two[1]]['area']
                if sz1 / sz2 > 1.5625 or sz1 / sz2 < 0.64:
                    # the two parts do not have comparable sizes
                    this_cell_is_pair = 0
                else:
                    this_cell_is_pair = 1
            else:
                # not far apart enough
                this_cell_is_pair = 0

        return this_cell_is_pair

    @staticmethod
    def _single_cell_gen_one_fov(
        row_index: int,
        row: pd.Series,
        single_cell_dir: Path,
        overwrite: bool = False
    ) -> Union[SingleCellGenOneFOVResult, SingleCellGenOneFOVFailure]:
        # TODO: currently, overwrite flag is not working. 
        # need to think more on how to deal with overwrite
        ########################################
        # parameters
        ########################################
        # Don't use dask for image reading
        aicsimageio.use_dask(False)
        standard_res_qcb = 0.108

        log.info(f"ready to process FOV: {row.FOVId}")

        ########################################
        # load image and segmentation
        ########################################
        try:
            [raw_fn, raw_mem0, raw_nuc0, raw_struct0, mem_seg_whole, 
             nuc_seg_whole, struct_seg_whole] = \
                PrepAnalysisSingleCellDs._load_image_and_seg(row)
        except (AssertionError, Exception) as e:
            log.info(
                f"Failed single cell generation for FOVId: {row.FOVId}. Error: {e}"
            )
            log.error(e, exc_info=True)
            return SingleCellGenOneFOVFailure(row.FOVId, True, str(e))

        log.info(f"Raw image and segmentation load successfully: {row.FOVId}")

        #########################################
        # run single cell qc in this fov
        #########################################
        try:
            [full_fov_pass, valid_cell] = \
                PrepAnalysisSingleCellDs._single_cell_qc_in_one_fov(
                    mem_seg_whole,
                    nuc_seg_whole,
                    row)
        except AssertionError as e:
            log.info(
                f"Skip single cell generation for FOVId: {row.FOVId}. Error: {e}"
            )
            # this is acceptable failure, set bug_flag=False
            return SingleCellGenOneFOVFailure(row.FOVId, False, str(e))
        except Exception as e:
            log.info(
                f"Failed single cell generation for FOVId: {row.FOVId}. Error: {e}"
            )
            log.error(e, exc_info=True)
            return SingleCellGenOneFOVFailure(row.FOVId, True, str(e))

        log.info(f"single cell QC done in FOV: {row.FOVId}")

        try:
            #################################################################
            # resize the image into isotropic dimension
            #################################################################
            raw_nuc = resize(raw_nuc0, (row.PixelScaleZ / standard_res_qcb, 
                                        row.PixelScaleY / standard_res_qcb,
                                        row.PixelScaleX / standard_res_qcb),
                             method='bilinear').astype(np.uint16)

            raw_mem = resize(raw_mem0, (row.PixelScaleZ / standard_res_qcb,
                                        row.PixelScaleY / standard_res_qcb,
                                        row.PixelScaleX / standard_res_qcb),
                             method='bilinear').astype(np.uint16)

            raw_str = resize(raw_struct0, (row.PixelScaleZ / standard_res_qcb,
                                           row.PixelScaleY / standard_res_qcb,
                                           row.PixelScaleX / standard_res_qcb),
                             method='bilinear').astype(np.uint16)

            mem_seg_whole = resize_to(mem_seg_whole, raw_mem.shape, method='nearest')
            nuc_seg_whole = resize_to(nuc_seg_whole, raw_nuc.shape, method='nearest')
            struct_seg_whole = resize_to(struct_seg_whole, raw_str.shape, 
                                         method='nearest')

            #################################################################
            # calculate fov related info
            #################################################################

            [index_to_cellid_map,
             cellid_to_index_map,
             index_to_centroid_map,
             stack_min_z,
             stack_max_z,
             true_edge_cells] = PrepAnalysisSingleCellDs._calculate_fov_info(
                row,
                valid_cell, 
                nuc_seg_whole,
                mem_seg_whole,
                raw_fn)

            #################################################################
            # calculate a dictionary to store FOV info
            #################################################################
            df_fov_meta = {
                'FOVId': row.FOVId,
                'structure_name': row.Gene,
                'position': row.ColonyPosition,
                'raw_fn': raw_fn,
                'str_filename': row.StructureSegmentationReadPath,
                'mem_seg_fn': row.MembraneSegmentationReadPath,
                'nuc_seg_fn': row.NucleusSegmentationReadPath,
                'index_to_id_dict': [index_to_cellid_map],
                'id_to_index_dict': [cellid_to_index_map],
                'xy_res': row.PixelScaleX,
                'z_res': row.PixelScaleZ,
                'stack_min_z': stack_min_z,
                'stack_max_z': stack_max_z,
                'scope_id': row.InstrumentId,
                'well_id': row.WellId,
                'well_name': row.WellName,
                'plateId': row.PlateId,
                'passage': row.Passage,
                'image_size': [list(raw_mem.shape)],
                'fov_seg_pass': full_fov_pass
            }

            log.info(f"FOV info is done: {row.FOVId}, ready to loop through cells")

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
                this_cell_nbr_candiate_list = list(np.unique(whole_template[
                    single_mem_dilate > 0]))
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
                    nuc_dist_3d = euc_dist_3d(index_to_centroid_map[nbr_id], 
                                              index_to_centroid_map[this_cell_index])
                    nuc_dist_2d = euc_dist_2d(index_to_centroid_map[nbr_id],
                                              index_to_centroid_map[this_cell_index])
                    overlap = overlap_area(mem_seg, mem_seg_whole == nbr_id)
                    this_cell_nbr_dist_3d.append((index_to_cellid_map[nbr_id],
                                                  nuc_dist_3d))
                    this_cell_nbr_dist_2d.append((index_to_cellid_map[nbr_id],
                                                  nuc_dist_2d))
                    this_cell_nbr_overlap_area.append((index_to_cellid_map[nbr_id], 
                                                       overlap))
                if len(this_cell_nbr_dist_3d) == 0:
                    this_cell_nbr_complete = 0

                # get cell id
                cell_id = index_to_cellid_map[this_cell_index]

                # make the path for saving single cell crop result
                thiscell_path = single_cell_dir / Path(str(cell_id))
                if os.path.isdir(thiscell_path):
                    rmtree(thiscell_path)
                os.mkdir(thiscell_path)

                [roi, nuc_seg, mem_seg, mem_top_mask_dilate, 
                 str_seg_crop, str_seg_crop_roof] = \
                    PrepAnalysisSingleCellDs._get_roi_and_crop(
                    mem_seg,
                    nuc_seg,
                    struct_seg_whole
                )

                # merge and save the cropped segmentation
                all_seg = np.stack([
                    nuc_seg,
                    mem_seg,
                    mem_top_mask_dilate,
                    str_seg_crop,
                    str_seg_crop_roof], axis=0)
                all_seg = np.expand_dims(np.transpose(all_seg, (1, 0, 2, 3)), axis=0)

                crop_seg_path = thiscell_path / 'segmentation.ome.tif'
                writer = save_tif.OmeTiffWriter(crop_seg_path)
                writer.save(all_seg)

                # crop raw image
                raw_nuc_thiscell = raw_nuc[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
                raw_mem_thiscell = raw_mem[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
                raw_str_thiscell = raw_str[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
                crop_raw_merged = np.expand_dims(np.stack((
                    raw_nuc_thiscell,
                    raw_mem_thiscell,
                    raw_str_thiscell), axis=1), axis=0)

                crop_raw_path = thiscell_path / 'raw.ome.tif'
                writer = save_tif.OmeTiffWriter(crop_raw_path)
                writer.save(crop_raw_merged)   

                # check for pair
                this_cell_is_pair = PrepAnalysisSingleCellDs._check_if_pair(nuc_seg)

                name_dict = {
                    "crop_raw": ['dna', 'membrane', 'structure'],
                    "crop_seg": ['dna_segmentation', 'membrane_segmentation',
                                 'membrane_segmentation', 'struct_segmentation',
                                 'struct_segmentation']
                }

                # out for mitotic classifier
                img_out = build_one_cell_for_classification(crop_raw_merged, mem_seg)
                out_fn = thiscell_path / 'for_mito_prediction.npy'
                np.save(out_fn, img_out)

                #########################################
                if len(true_edge_cells) > 0 and (this_cell_index in true_edge_cells):
                    this_is_edge_cell = 1
                else:
                    this_is_edge_cell = 0

                # write qcb cell meta
                df_cell_meta.append({
                    'CellId': cell_id,
                    'structure_name': row.Gene,
                    'pair': this_cell_is_pair,
                    'this_cell_nbr_complete': this_cell_nbr_complete,
                    'this_cell_nbr_dist_3d': [this_cell_nbr_dist_3d],
                    'this_cell_nbr_dist_2d': [this_cell_nbr_dist_2d],
                    'this_cell_nbr_overlap_area': [this_cell_nbr_overlap_area],
                    'roi': [roi],
                    'crop_raw': crop_raw_path,
                    'crop_seg': crop_seg_path,
                    'name_dict': [name_dict],
                    'scale_micron': [[0.108, 0.108, 0.108]],
                    'edge_flag': this_is_edge_cell,
                    'fov_id': row.FOVId,
                    'fov_path': raw_fn,
                    'stack_min_z': stack_min_z,
                    'stack_max_z': stack_max_z,
                    'image_size': [list(raw_mem.shape)],
                    'plateId': row.PlateId,
                    'position': row.ColonyPosition,
                    'scope_id': row.InstrumentId,
                    'well_id': row.WellId,
                    'well_name': row.WellName,
                    'passage': row.Passage
                })

            #  single cell generation succeeds in this FOV
            return SingleCellGenOneFOVResult(row.FOVId, df_fov_meta, df_cell_meta)

        except Exception as e:
            log.info(
                f"Failed single cell generation for FOVId: {row.FOVId}. Error: {e}"
            )
            log.error(e, exc_info=True)
            return SingleCellGenOneFOVFailure(row.FOVId, True, str(e))

    @log_run_params
    def run(
        self,
        dataset: Union[str, Path, pd.DataFrame, dd.DataFrame],
        distributed_executor_address: Optional[str] = None,
        save_fov_dataset: bool = True,
        debug: bool = False,
        overwrite: bool = False,
        **kwargs
    ) -> List[Path]:
        """
        Run single cell generation, which will be used for analysis

        Protected Parameters
        --------------------
        distributed_executor_address: Optional[str]
            An optional executor address to pass to some computation engine.
        debug: bool
            A debug flag for the developer to use to manipulate how much data runs,
            how it is processed, etc.
            Default: False (Do not debug)
        overwrite: bool
            If this step has already partially or completely run, should it overwrite
            the previous files or not.
            Default: False (Do not overwrite or regenerate files)

        Parameters
        ----------
        dataset: Union[str, Path, pd.DataFrame, dd.DataFrame]
            The dataset to use for generating single cell dataset.

            **Required dataset columns:** *["FOVId", "SourceReadPath",
            "NucleusSegmentationReadPath", "MembraneSegmentationReadPath",
            "ChannelIndexDNA", "ChannelIndexMembrane", "ChannelIndexStructure",
            "ChannelIndexBrightfield"]*
        save_fov_data: bool
            A flag for saving fov dataset or not, after preparation
            Default: True (save fov dataset to csv)

        Returns
        -------
        manifest_save_path: Path
            Path to the produced manifest of single cell dataset
        """
        # Your code here
        #
        # The `self.step_local_staging_dir` is exposed to save files in
        #
        # The user should set `self.manifest` to a dataframe of absolute paths that
        # point to the created files and each files metadata
        #
        # By default, `self.filepath_columns` is ["filepath"], but should be edited
        # if there are more than a single column of filepaths
        #
        # By default, `self.metadata_columns` is [], but should be edited to include
        # any columns that should be parsed for metadata and attached to objects
        #
        # The user should not rely on object state to retrieve results from prior steps.
        # I.E. do not call use the attribute self.upstream_tasks to retrieve data.
        # Pass the required path to a directory of files, the path to a prior manifest,
        # or in general, the exact parameters required for this function to run.

        # Handle dataset provided as string or path
        if isinstance(dataset, (str, Path)):
            dataset = pq.read_table(Path(dataset).expanduser().resolve(strict=True))
            dataset = dataset.to_pandas()

        # HACK: for some reason the structure segmentation read path is empty
        # HACK: temporary solution, use membrane seg as fake structure seg
        dataset['StructureSegmentationReadPath'] = \
            dataset['MembraneSegmentationReadPath']

        # HACK: AlignmentReadPath should exist in final query
        if 'AlignedImageReadPath' not in dataset.columns:
            dataset = dataset.assign(AlignedImageReadPath=None)

        # create a fov data frame
        fov_dataset = dataset.copy()
        fov_dataset.drop_duplicates(subset=["FOVId"], keep="first", inplace=True)
        fov_dataset.drop(["CellId", "CellIndex"], axis=1, inplace=True)

        # add two new colums
        fov_dataset['index_to_id_dict'] = np.empty((len(fov_dataset), 0)).tolist()
        fov_dataset['id_to_index_dict'] = np.empty((len(fov_dataset), 0)).tolist()

        for row in fov_dataset.itertuples():
            df_one_fov = dataset.query("FOVId==@row.FOVId")

            # collect all cells from this fov, and create mapping
            fov_index_to_id_dict = dict()
            fov_id_to_index_dict = dict()
            for cell_row in df_one_fov.itertuples():
                fov_index_to_id_dict[cell_row.CellIndex] = cell_row.CellId
                fov_id_to_index_dict[cell_row.CellId] = cell_row.CellIndex
            # add dictioinary back to fov dataframe
            fov_dataset.at[row.Index, 'index_to_id_dict'] = [fov_index_to_id_dict]
            fov_dataset.at[row.Index, 'id_to_index_dict'] = [fov_id_to_index_dict]

        # Log original length of cell dataset
        log.info(f"Original dataset length: {len(dataset)}")

        # Create single cell directory
        single_cell_dir = self.step_local_staging_dir / "single_cells"
        single_cell_dir.mkdir(exist_ok=True)
        log.info(f"single cells will be saved into: {single_cell_dir}")

        ### for debug ###
        # for ridx, row in fov_dataset.iterrows():
        #     x = self._single_cell_gen_one_fov(ridx, row, single_cell_dir, overwrite)

        # from concurrent.futures import ProcessPoolExecutor
        # with ProcessPoolExecutor() as exe:
        #    results = exe.map(
        #        self._single_cell_gen_one_fov,
        #        *zip(*list(fov_dataset.iterrows())),
        #        [single_cell_dir for i in range(len(dataset))],
        #        [overwrite for i in range(len(dataset))]
        #    )

        # Process each row
        with DistributedHandler(distributed_executor_address) as handler:
            # Start processing
            results = handler.batched_map(
                self._single_cell_gen_one_fov,
                # Convert dataframe iterrows into two lists of items to iterate over
                # One list will be row index
                # One list will be the pandas series of every row
                *zip(*list(fov_dataset.iterrows())),
                # Pass the other parameters as list of the same thing for each
                # mapped function call
                [single_cell_dir for i in range(len(fov_dataset))],
                [overwrite for i in range(len(fov_dataset))],
                batch_size=10,
            )

        # Generate fov paths rows
        fov_meta_gather = []
        cell_meta_gather = []
        errors = []
        bad_data = []
        for result in results:
            if isinstance(result, SingleCellGenOneFOVResult):
                fov_meta_gather.append(result.fov_meta)
                cell_meta_gather.append(result.cell_meta)
            elif isinstance(result, SingleCellGenOneFOVFailure):
                if result.bug_fail:
                    errors.append(
                        {DatasetFields.FOVId: result.fov_id, "Error": result.error}
                    )
                else:
                    bad_data.append(
                        {DatasetFields.FOVId: result.fov_id, "Error": result.error}
                    )

        # save fov datasets
        final_fov_meta = pd.DataFrame(fov_meta_gather)
        fov_manifest_save_path = self.step_local_staging_dir / "fov_dataset.csv"
        final_fov_meta.to_csv(fov_manifest_save_path, index=False)

        # build output datasets
        final_cell_meta = pd.DataFrame(list(itertools.chain(*cell_meta_gather)))
        self.manifest = final_cell_meta
        cell_manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.final_cell_meta.to_csv(cell_manifest_save_path, index=False)

        # Save errored FOVs to JSON
        with open(self.step_local_staging_dir / "errors.json", "w") as write_out:
            json.dump(errors, write_out)

        with open(self.step_local_staging_dir / "bad_data.json", "w") as write_out:
            json.dump(bad_data, write_out)

        return [fov_manifest_save_path, cell_manifest_save_path]
