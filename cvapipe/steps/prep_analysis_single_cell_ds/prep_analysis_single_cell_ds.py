#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import dask.dataframe as dd
import pandas as pd
import numpy as np
import aicsimageio
from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer as save_tif

from aics_dask_utils import DistributedHandler
from datastep import Step, log_run_params

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class SingleCellGenOneFOVResult(NamedTuple):
    fov_id: Union[int, str]
    path: Path


class SingleCellGenOneFOVFailure(NamedTuple):
    fov_id: int
    fail_type: str


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
        # Don't use dask for image reading
        aicsimageio.use_dask(False)

        ###############################
        # TODO: add to constants.py
        standard_res_qcb = 0.108
        dist_cutoff = 85
        mem_cutoff = 2000
        nuc_cutoff = 1200
        ###############################

        # get file paths (pipeline 4.4 needs to use aligned images
        # but, pipline 4.0-4.3 does not)
        if row.AlignedImageReadPath is None:
            raw_fn = row.SourceReadPath
        else:
            raw_fn = row.AlignedImageReadPath
        assert os.path.exists(raw_fn), 'original image cannot be found'

        # get cell/nuclear segmentation
        # because the results are save in one file, MembraneSegmentationFilename
        # and NucleusSegmentationFilename should be the same
        assert row.MembraneSegmentationFilename == row.NucleusSegmentationFilename,\
            'MembraneSegmentationFilename and NucleusSegmentationFilename do not match'
        combo_seg_fn = row.MembraneSegmentationReadPath
        assert os.path.exists(combo_seg_fn), 'cell segmentation cannot be found'


        # get structure segmentation (only one version)
        str_fn = row.StructureSegmentationReadPath
        assert os.path.exists(str_fn), 'structure segmentation cannot be found'

        str_reader = AICSImage(str_fn)
        str_seg = np.squeeze(str_reader.data)

        # load resolution
        # TODO: load from query
        xy_res = float(row['PixelScaleX'])
        z_res = float(row['PixelScaleZ'])

        # load other important meta data
        imaging_position = row['ColonyPosition']
        fov_id = row['FOVId']
        scope_id = row['InstrumentId']
        well_id = row['WellId']
        well_name = row['WellName']
        plateId = row['PlateId']

        '''
        # load mem/dna segmentation
        mem_reader = AICSImage(mem_fn)
        mem_seg_whole = mem_reader.data[0,0,:,:,:]
        nuc_reader = AICSImage(nuc_fn)
        nuc_seg_whole = nuc_reader.data[0,0,:,:,:]
        assert mem_seg_whole.shape == nuc_seg_whole.shape
        '''

        seg_reader = AICSImage(combo_seg_fn)
        mem_seg_whole = seg_reader.data[0, 1, :, :, :]
        nuc_seg_whole = seg_reader.data[0, 0, :, :, :]

        # no single bd_cell_index in new version
        # bd_cell_index = mem_seg_whole.max()

        #################################################################
        # make sure the segmentation of image is not failed terribly
        ##################################################################

        # double check catastrophic failure, quick reject
        if mem_seg_whole.max() < 4 or nuc_seg_whole.max() < 4:
            print('very few cells left .... skip')
            sys.exit(0)

        # remove too small cells
        all_cell_index_list = list(np.unique(mem_seg_whole[mem_seg_whole > 0]))
        maybe_okay_seg = 0
        for this_cell_index in all_cell_index_list:
            if np.count_nonzero(mem_seg_whole == this_cell_index) < 70000 \
            or np.count_nonzero(nuc_seg_whole == this_cell_index) < 10000:
                # TODO: change to constant
                mem_seg_whole[mem_seg_whole == this_cell_index] = 0
                nuc_seg_whole[nuc_seg_whole == this_cell_index] = 0
            else:
                maybe_okay_seg += 1
        if maybe_okay_seg < 4:
            sys.exit(0)

        # load raw image 
        raw_reader = AICSImage(raw_fn)

        ChannelNumber405 = row['ChannelNumber405']
        ChannelNumber638 = row['ChannelNumber638']
        ChannelNumberStruct = row['ChannelNumberStruct']
        # ChannelNumberBrightfield = row['ChannelNumberBrightfield']

        raw_mem0 = raw_reader.data[0, ChannelNumber638, :, :, :]
        raw_nuc0 = raw_reader.data[0, ChannelNumber405, :, :, :]
        raw_str0 = raw_reader.data[0, ChannelNumberStruct, :, :, :]

        # prune the results (remove cells touching image boundary)
        boundary_mask = np.zeros_like(mem_seg_whole)
        boundary_mask[:, :3, :] = 1
        boundary_mask[:, -3:, :] = 1
        boundary_mask[:, :, :3] = 1
        boundary_mask[:, :, -3:] = 1
        bd_idx = list(np.unique(mem_seg_whole[boundary_mask > 0]))
        all_cell_index_list = list(np.unique(mem_seg_whole[mem_seg_whole > 0]))
        valid_cell = list(set(all_cell_index_list) - set(bd_idx))

        ##########################################################################
        # resize the image for cropping
        ##########################################################################
        raw_nuc = resize(raw_nuc0, (z_res / standard_res_qcb, 
                                    xy_res / standard_res_qcb,
                                    xy_res / standard_res_qcb),
                        method='cubic')
        raw_nuc = raw_nuc.astype(np.uint16)

        raw_mem = resize(raw_mem0, (z_res / standard_res_qcb,
                                    xy_res / standard_res_qcb,
                                    xy_res / standard_res_qcb),
                        method='cubic')
        raw_mem = raw_mem.astype(np.uint16)

        raw_str = resize(raw_str0, (z_res / standard_res_qcb,
                                    xy_res / standard_res_qcb,
                                    xy_res / standard_res_qcb),
                        method='cubic')
        raw_str = raw_str.astype(np.uint16)

        mem_seg_whole = resize_to(mem_seg_whole, raw_mem.shape, method='nearest')
        nuc_seg_whole = resize_to(nuc_seg_whole, raw_nuc.shape, method='nearest')
        str_seg = resize_to(str_seg, raw_str.shape, method='nearest')

        # automatic single cell qc and set flag for full fov pass/fail
        full_fov_pass = 1
        valid_cell_0 = valid_cell.copy()
        for list_idx, this_cell_index in enumerate(valid_cell):
            # check cell size
            single_mem = mem_seg_whole == this_cell_index
            if np.count_nonzero(single_mem > 0) < mem_cutoff:
                valid_cell_0.remove(this_cell_index)
                full_fov_pass = 0
                continue

            # check nucleus size
            single_nuc = nuc_seg_whole == this_cell_index
            if np.count_nonzero(single_nuc > 0) < nuc_cutoff:
                valid_cell_0.remove(this_cell_index)
                full_fov_pass = 0
                continue

            # make sure the cell is not leaking to the bottom or top
            z_range_single = np.where(np.any(single_mem, axis=(1, 2)))
            z_range_single = z_range_single[0]
            single_min_z = z_range_single[0]
            single_max_z = z_range_single[-1]

            if single_min_z == 0 or single_max_z >= single_mem.shape[0] - 1:
                valid_cell_0.remove(this_cell_index)
                full_fov_pass = 0
                continue

        valid_cell = valid_cell_0
        if len(valid_cell) < 2:  # if only one cell left or no cell left, just throw it away
            print('very few cells left ...')
            sys.exit(0)

        # identify index to CellId mapping
        index_to_cellid_mapping = dict()
        cellid_to_index_mapping = dict()
        for list_idx, this_cell_index in enumerate(valid_cell):
            index_to_cellid_mapping[this_cell_index] = str(uuid.uuid4())
        for index_dict, cellid_dict in index_to_cellid_mapping.items():
            cellid_to_index_mapping[cellid_dict] = index_dict 

        # compute center of mass
        index_to_centroid_mapping = dict()
        center_list = center_of_mass(nuc_seg_whole > 0, nuc_seg_whole, valid_cell)
        for list_idx, this_cell_index in enumerate(valid_cell):
            index_to_centroid_mapping[this_cell_index] = center_list[list_idx]

        # computer whole stack min/max z
        mem_seg_whole_valid = np.zeros_like(mem_seg_whole)
        for list_idx, this_cell_index in enumerate(valid_cell):
            mem_seg_whole_valid[mem_seg_whole == this_cell_index] = this_cell_index
        z_range_whole = np.where(np.any(mem_seg_whole_valid, axis=(1, 2)))
        z_range_whole = z_range_whole[0]
        stack_min_z = z_range_whole[0]
        stack_max_z = z_range_whole[-1]

        # build fov meta entry
        full_fov_size = raw_mem.shape

        '''
        df_fov = pd.DataFrame({'FOVId':fov_id,'structure_name':args.ds,'position':imaging_position, \
                'raw_fn':raw_fn,'str_filenames':[str_filenames],'str_seg_namelist':[str_seg_namelist],'mem_seg_fn':combo_seg_fn,'nuc_seg_fn':combo_seg_fn, \
                'index_to_id_dict':[index_to_cellid_mapping], \
                'id_to_index_dict':[cellid_to_index_mapping],'xy_res':xy_res,'z_res':z_res, 'stack_min_z':stack_min_z, \
                'stack_max_z':stack_max_z, 'scope_id':scope_id, 'well_id':well_id, 'well_name':well_name, 'plateId':plateId, 'image_size': [list(full_fov_size)],'fov_seg_pass':full_fov_pass}, \
                columns=['FOVId','structure_name','position','raw_fn','str_filenames','str_seg_namelist','mem_seg_fn','nuc_seg_fn', \
                'index_to_id_dict','id_to_index_dict','xy_res','z_res','stack_min_z','stack_max_z','scope_id', \
                'well_id','well_name','plateId','image_size','fov_seg_pass'],index=[1])

        with open(fov_csv, 'w') as f:
            df_fov.to_csv(f, header=True, index=False) 
        '''

        # find true edge cells
        true_edge_cells = []
        reg = re.compile('(-|_)((\d)?)(e)((\d)?)(-|_)')
        # check if edge fov
        if reg.search(os.path.basename(raw_fn)):
            true_edge_cells = find_true_edge_cells(mem_seg_whole_valid)

        # loop through all valid cells in this fov
        full_fov_valid_flag = True
        for list_idx, this_cell_index in enumerate(valid_cell):
            nuc_seg = nuc_seg_whole == this_cell_index
            mem_seg = mem_seg_whole == this_cell_index

            ###########################
            # implement nbr info
            ###########################
            single_mem_dilate = dilation(mem_seg, selem=ball(3))
            whole_template = mem_seg_whole.copy()
            whole_template[mem_seg == this_cell_index] = 0
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

                nuc_dist_3d = euc_distance_3d(index_to_centroid_mapping[nbr_id], 
                                            index_to_centroid_mapping[this_cell_index])
                nuc_dist_2d = euc_distance_2d(index_to_centroid_mapping[nbr_id],
                                            index_to_centroid_mapping[this_cell_index])
                overlap = overlap_area(mem_seg, mem_seg_whole == nbr_id)
                this_cell_nbr_dist_3d.append((index_to_cellid_mapping[nbr_id], nuc_dist_3d))
                this_cell_nbr_dist_2d.append((index_to_cellid_mapping[nbr_id], nuc_dist_2d))
                this_cell_nbr_overlap_area.append((index_to_cellid_mapping[nbr_id], 
                                                overlap))
                if len(this_cell_nbr_dist_3d) == 0:
                    this_cell_nbr_complete = 0

            # generate cell id
            cell_id = index_to_cellid_mapping[this_cell_index]

            # make the path for saving single cell crop result
            thiscell_path = parent_path + ds + '/single_cells/' + str(cell_id)
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

            # extra +2 for z top
            roi = [max(z_range[0] - 10, 0), min(z_range[-1] + 12, mem_seg.shape[0]),
                max(y_range[0] - 80, 0), min(y_range[-1] + 80, mem_seg.shape[1]),
                max(x_range[0] - 80, 0), min(x_range[-1] + 80, mem_seg.shape[2])]

            # roof augmentation
            mem_nearly_top_z = int(z_range[0] + round(0.75 
                                                    * (z_range[-1] - z_range[0] + 1)))
            mem_top_mask = np.zeros(mem_seg.shape, dtype=np.byte)
            mem_top_mask[mem_nearly_top_z:, :, :] = mem_seg[mem_nearly_top_z:, :, :] > 0                  
            mem_top_mask_dilate = dilation(mem_top_mask > 0, selem=np.ones((21, 1, 1),
                                        dtype=np.byte))
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
            str_seg_crop_list = []
            for si in range(len(str_seg_list)):

                str_seg = str_seg_list[si].copy()
                str_seg = str_seg[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
                str_seg[mem_seg < 1] = 0

                str_seg = str_seg.astype(np.uint8)
                str_seg[str_seg > 0] = 255
                str_seg_crop_list.append(str_seg)

            # crop str seg (with roof augmentation)
            for si in range(len(str_seg_list)):

                str_seg = str_seg_list[si].copy()
                str_seg = str_seg[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
                str_seg[mem_top_mask_dilate < 1] = 0

                str_seg = str_seg.astype(np.uint8)
                str_seg[str_seg > 0] = 255
                str_seg_crop_list.append(str_seg)

            # save all the cropped segmentation
            all_seg = np.concatenate([np.stack([nuc_seg, mem_seg, mem_top_mask_dilate],
                                            axis=0),
                                    np.stack(str_seg_crop_list, axis=0)], axis=0)
            all_seg = np.expand_dims(np.transpose(all_seg, (1, 0, 2, 3)), axis=0)

            crop_seg_path = thiscell_path + os.sep + 'segmentation.ome.tif'
            writer = omeTifWriter.OmeTifWriter(crop_seg_path)
            writer.save(all_seg)

            # crop raw image
            raw_nuc_thiscell = raw_nuc[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
            raw_mem_thiscell = raw_mem[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
            raw_str_thiscell = raw_str[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
            crop_raw_merged = np.expand_dims(np.stack((raw_nuc_thiscell, raw_mem_thiscell,
                                                    raw_str_thiscell), axis=1), axis=0)

            crop_raw_path = thiscell_path + os.sep + 'raw.ome.tif'
            writer = omeTifWriter.OmeTifWriter(crop_raw_path)
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
                dis = euc_distance_3d(stats[large_two[0]]['centroid'], 
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

            # out
            img_out = build_one_cell_for_classification(crop_raw_merged, mem_seg)
            out_fn = thiscell_path + os.sep + 'for_mito_prediction.npy'
            np.save(out_fn, img_out)

            #########################################
            if len(true_edge_cells) > 0 and (this_cell_index in true_edge_cells):
                this_is_edge_cell = 1
            else:
                this_is_edge_cell = 0
            '''
            # write qcb cell meta
            df = pd.DataFrame({'CellId':cell_id,'structure_name':args.ds,\
                'mitosis_2':0,'mitosis_5':0,'mitosis_8':0,'pair':this_cell_is_pair,'this_cell_nbr_complete':this_cell_nbr_complete,'this_cell_nbr_dist_3d':[this_cell_nbr_dist_3d],\
                'this_cell_nbr_dist_2d':[this_cell_nbr_dist_2d], 'this_cell_nbr_overlap_area':[this_cell_nbr_overlap_area],\
                'position':imaging_position,'crop_raw':crop_raw_path,'crop_seg':crop_seg_path,'name_dict':[name_dict], \
                'roi':[roi], 'scale_micron':[[0.108,0.108,0.108]] ,'fov_id':fov_id,'fov_path':raw_fn,\
                'stack_min_z':stack_min_z, 'stack_max_z':stack_max_z,'image_size': [list(full_fov_size)],\
                'scope_id':scope_id, 'well_id':well_id, 'well_name':well_name, 'edge_flag':this_is_edge_cell}, \
                columns=['CellId','structure_name','mitosis_2','mitosis_5','mitosis_8','pair','this_cell_nbr_complete','this_cell_nbr_dist_3d', 'this_cell_nbr_dist_2d',\
                'this_cell_nbr_overlap_area','position','crop_raw','crop_seg','name_dict','roi','scale_micron','fov_id','fov_path','stack_min_z','stack_max_z','image_size',\
                'scope_id', 'well_id','well_name','edge_flag'],index=[1])
            if not os.path.exists(output_csv):
                with open(output_csv, 'w') as f:
                    df.to_csv(f, header=True, index=False) 
            else:
                with open(output_csv, 'a') as f:
                    df.to_csv(f, header=False, index=False)
            ''' 


    @log_run_params
    def run(
        self,
        dataset: Union[str, Path, pd.DataFrame, dd.DataFrame],
        single_cell_path: Path = "single_cells",
        distributed_executor_address: Optional[str] = None,
        clean: bool = False,
        debug: bool = False,
        **kwargs
    ) -> Path:
        """
        Run single cell generation, which will be used for analysis

        Protected Parameters
        --------------------
        distributed_executor_address: Optional[str]
            An optional executor address to pass to some computation engine.
        clean: bool
            Should the local staging directory be cleaned prior to this run.
            Default: False (Do not clean)
        debug: bool
            A debug flag for the developer to use to manipulate how much data runs,
            how it is processed, etc.
            Default: False (Do not debug)

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
                *zip(*list(fov_dataset.iterrows())),
                # Pass the other parameters as list of the same thing for each
                # mapped function call
                [desired_pixel_sizes for i in range(len(fov_dataset))],
                [fovs_dir for i in range(len(fov_dataset))],
                [overwrite for i in range(len(dataset))],
            )
            results = handler.gather(futures)

        return
