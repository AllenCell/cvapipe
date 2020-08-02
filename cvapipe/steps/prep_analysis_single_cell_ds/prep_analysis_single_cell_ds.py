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
import re
from shutil import rmtree
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
    def _single_cell_gen_one_fov(
        row_index: int,
        row: pd.Series,
        single_cell_dir: Path,
        overwrite: bool,
    ) -> Union[SingleCellGenOneFOVResult, SingleCellGenOneFOVFailure]:
        # TODO: currently, overwrite flag is not working. 
        # need to think more on how to deal with overwrite

        # Don't use dask for image reading
        aicsimageio.use_dask(False)

        ###############################
        standard_res_qcb = 0.108
        dist_cutoff = 85
        min_mem_size = 70000
        min_nuc_size = 10000
        ###############################

        ################################################################
        # Part 1: load images and segmentations
        ################################################################

        # select proper raw file to use
        # (pipeline 4.4 needs to use aligned images, but pipline 4.0-4.3 does not)
        if row.AlignedImageReadPath is None:
            raw_fn = row.SourceReadPath
        else:
            raw_fn = row.AlignedImageReadPath

        # verify filepaths 
        try:
            assert os.path.exists(raw_fn), f"original image not found: {raw_fn}"
            assert os.path.exists(row.MembraneSegmentationReadPath),\
                f"cell segmentation not found: {row.MembraneSegmentationReadPath}"
            assert os.path.exists(row.StructureSegmentationReadPath),\
                f"structure segmentation not found {row.StructureSegmentationReadPath}"
        except AssertionError as e:
            log.info(
                f"Failed single cell generation for FOVId: {row.FOVId}. Error: {e}"
            )
            return SingleCellGenOneFOVFailure(row.FOVId, True, str(e))

        # get the raw image and split into different channels
        raw_reader = AICSImage(raw_fn)
        raw_mem0 = raw_reader.get_image_data("ZYX", S=0, T=0, C=row.ChannelNumber638) 
        raw_nuc0 = raw_reader.get_image_data("ZYX", S=0, T=0, C=row.ChannelNumber405) 
        raw_str0 = raw_reader.get_image_data("ZYX", S=0, T=0, C=row.ChannelNumberStruct)

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
        str_seg = np.squeeze(imread(row.StructureSegmentationReadPath))

        #################################################################
        # part 2: make sure the cell/nucleus segmentation not failed terribly
        # remove the bad cells when possible (e.g., very small cells)
        ##################################################################

        # flag for any segmented object in this FOV removed as bad cells
        full_fov_pass = 1
        try:
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
            valid_cell = list(full_set_with_no_boundary 
                              - set(row.index_to_id_dict.keys()))

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
            assert len(valid_cell_0) > 1, \
                f"very few cells left after single cell QC in {row.FOVId}"

            # assign the tempory list variable back
            valid_cell = valid_cell_0

        except AssertionError as e:
            log.info(
                f"Failed single cell generation for FOVId: {row.FOVId}. Error: {e}"
            )
            return SingleCellGenOneFOVFailure(row.FOVId, False, str(e))
        except Exception as e:
            log.info(
                f"Failed single cell generation for FOVId: {row.FOVId}. Error: {e}"
            )
            return SingleCellGenOneFOVFailure(row.FOVId, True, str(e))

        try:
            ##########################################################################
            # resize the image into isotropic dimension
            ##########################################################################
            raw_nuc = resize(raw_nuc0, (row.PixelScaleZ / standard_res_qcb, 
                                        row.PixelScaleY / standard_res_qcb,
                                        row.PixelScaleX / standard_res_qcb),
                             method='bilinear').astype(np.uint16)

            raw_mem = resize(raw_mem0, (row.PixelScaleZ / standard_res_qcb,
                                        row.PixelScaleY / standard_res_qcb,
                                        row.PixelScaleX / standard_res_qcb),
                             method='bilinear').astype(np.uint16)

            raw_str = resize(raw_str0, (row.PixelScaleZ / standard_res_qcb,
                                        row.PixelScaleY / standard_res_qcb,
                                        row.PixelScaleX / standard_res_qcb),
                             method='bilinear').astype(np.uint16)

            mem_seg_whole = resize_to(mem_seg_whole, raw_mem.shape, method='nearest')
            nuc_seg_whole = resize_to(nuc_seg_whole, raw_nuc.shape, method='nearest')
            str_seg = resize_to(str_seg, raw_str.shape, method='nearest')

            # identify index and CellId 1:1 mapping
            index_to_cellid_map = dict()
            cellid_to_index_map = dict()
            for list_idx, this_cell_index in enumerate(valid_cell):
                # this is always valid since indices not in index_to_id_dict.keys() 
                # have been removed
                index_to_cellid_map[this_cell_index] = \
                    row.index_to_id_dict[this_cell_index]
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
                thiscell_path = single_cell_dir / str(cell_id)
                if os.path.isdir(thiscell_path):
                    rmtree(thiscell_path)
                os.mkdir(thiscell_path)

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
                str_seg_crop = str_seg[roi[0]:roi[1], roi[2]:roi[3], 
                                       roi[4]:roi[5]].astype(np.uint8)
                str_seg_crop[mem_seg < 1] = 0
                str_seg_crop[str_seg_crop > 0] = 255

                # crop str seg (with roof augmentation)
                str_seg_crop_roof = str_seg[roi[0]:roi[1], roi[2]:roi[3], 
                                            roi[4]:roi[5]].astype(np.uint8)
                str_seg_crop_roof[mem_top_mask_dilate < 1] = 0
                str_seg_crop_roof[str_seg_crop_roof > 0] = 255

                # save the cropped segmentation
                all_seg = np.stack([
                    nuc_seg,
                    mem_seg,
                    mem_top_mask_dilate,
                    str_seg_crop,
                    str_seg_crop_roof], axis=0),
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
            return SingleCellGenOneFOVFailure(row.FOVId, True, str(e))

    @log_run_params
    def run(
        self,
        dataset: Union[str, Path, pd.DataFrame, dd.DataFrame],
        single_cell_path: Path = "single_cells",
        distributed_executor_address: Optional[str] = None,
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
            dataset = pd.read_csv(Path(dataset).expanduser().resolve(strict=True))

        # Log original length of cell dataset
        log.info(f"Original dataset length: {len(dataset)}")

        # Create single cell directory
        single_cell_dir = self.step_local_staging_dir / single_cell_path
        single_cell_dir.mkdir(exist_ok=True)

        # Process each row
        with DistributedHandler(distributed_executor_address) as handler:
            # Start processing
            futures = handler.client.map(
                self._generate_standardized_fov_array,
                # Convert dataframe iterrows into two lists of items to iterate over
                # One list will be row index
                # One list will be the pandas series of every row
                *zip(*list(dataset.iterrows())),
                # Pass the other parameters as list of the same thing for each
                # mapped function call
                [single_cell_dir for i in range(len(dataset))],
                [overwrite for i in range(len(dataset))]
            )
            results = handler.gather(futures)

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

        # build output datasets
        final_fov_meta = pd.DataFrame(fov_meta_gather)
        final_cell_meta = pd.DataFrame(list(itertools.chain(*cell_meta_gather)))

        # Join original dataset to the fov paths
        self.fov_manifest = final_fov_meta
        self.cell_manifest = final_cell_meta

        # Save manifest to CSV
        fov_manifest_save_path = self.step_local_staging_dir / "fov_manifest.csv"
        self.fov_manifest.to_csv(fov_manifest_save_path, index=False)

        cell_manifest_save_path = self.step_local_staging_dir / "cell_manifest.csv"
        self.cell_manifest.to_csv(cell_manifest_save_path, index=False)

        # Save errored FOVs to JSON
        with open(self.step_local_staging_dir / "errors.json", "w") as write_out:
            json.dump(errors, write_out)

        with open(self.step_local_staging_dir / "bad_data.json", "w") as write_out:
            json.dump(bad_data, write_out)

        return [fov_manifest_save_path, cell_manifest_save_path]
