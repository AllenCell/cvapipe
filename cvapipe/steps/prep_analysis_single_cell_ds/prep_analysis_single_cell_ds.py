#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import dask.dataframe as dd
import pandas as pd
import numpy as np
import itertools
import pyarrow.parquet as pq
from aics_dask_utils import DistributedHandler
from datastep import Step, log_run_params

from ...constants import DatasetFields
from cvapipe.utils.prep_analysis_single_cell_utils import single_cell_gen_one_fov

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class PrepAnalysisSingleCellDs(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        dataset: Union[str, Path, pd.DataFrame, dd.DataFrame],
        distributed_executor_address: Optional[str] = None,
        save_fov_dataset: bool = True,
        debug: bool = False,
        overwrite: bool = False,
        **kwargs,
    ) -> Path:
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
        cell_manifest_save_path: Path
            Path to the produced manifest of single cell dataset
        """
        # Handle dataset provided as string or path
        if isinstance(dataset, (str, Path)):
            dataset = pq.read_table(Path(dataset).expanduser().resolve(strict=True))
            dataset = dataset.to_pandas()

        # HACK: for some reason the structure segmentation read path is empty
        # HACK: temporary solution, use membrane seg as fake structure seg
        dataset["StructureSegmentationReadPath"] = dataset[
            "MembraneSegmentationReadPath"
        ]

        # HACK: AlignmentReadPath should exist in final query
        if "AlignedImageReadPath" not in dataset.columns:
            dataset = dataset.assign(AlignedImageReadPath=None)

        # create a fov data frame
        fov_dataset = dataset.copy()
        fov_dataset.drop_duplicates(subset=["FOVId"], keep="first", inplace=True)
        fov_dataset.drop(["CellId", "CellIndex"], axis=1, inplace=True)

        # add two new colums
        fov_dataset["index_to_id_dict"] = np.empty((len(fov_dataset), 0)).tolist()
        fov_dataset["id_to_index_dict"] = np.empty((len(fov_dataset), 0)).tolist()

        for row in fov_dataset.itertuples():
            df_one_fov = dataset.query("FOVId==@row.FOVId")

            # collect all cells from this fov, and create mapping
            fov_index_to_id_dict = dict()
            fov_id_to_index_dict = dict()
            for cell_row in df_one_fov.itertuples():
                fov_index_to_id_dict[cell_row.CellIndex] = cell_row.CellId
                fov_id_to_index_dict[cell_row.CellId] = cell_row.CellIndex
            # add dictioinary back to fov dataframe
            fov_dataset.at[row.Index, "index_to_id_dict"] = [fov_index_to_id_dict]
            fov_dataset.at[row.Index, "id_to_index_dict"] = [fov_id_to_index_dict]

        # Log original length of cell dataset
        log.info(f"Original dataset length: {len(dataset)}")

        # Create single cell directory
        single_cell_dir = self.step_local_staging_dir / "single_cells"
        single_cell_dir.mkdir(exist_ok=True)
        log.info(f"single cells will be saved into: {single_cell_dir}")

        # Process each row
        with DistributedHandler(distributed_executor_address) as handler:
            # Start processing
            results = handler.batched_map(
                single_cell_gen_one_fov,
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
            if len(result) == 2:
                fov_meta_gather.append(result[0])
                cell_meta_gather.append(result[1])
            elif len(result) == 3:
                if result[1]:
                    errors.append({DatasetFields.FOVId: result[0], "Error": result[2]})
                else:
                    bad_data.append(
                        {DatasetFields.FOVId: result[0], "Error": result[2]}
                    )

        # save fov datasets
        final_fov_meta = pd.DataFrame(fov_meta_gather)
        fov_manifest_save_path = self.step_local_staging_dir / "fov_dataset.csv"
        final_fov_meta.to_csv(fov_manifest_save_path, index=False)

        # build output datasets
        final_cell_meta = pd.DataFrame(list(itertools.chain(*cell_meta_gather)))
        self.manifest = final_cell_meta
        cell_manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        final_cell_meta.to_csv(cell_manifest_save_path, index=False)

        # Save errored FOVs to JSON
        with open(self.step_local_staging_dir / "errors.json", "w") as write_out:
            json.dump(errors, write_out)

        with open(self.step_local_staging_dir / "bad_data.json", "w") as write_out:
            json.dump(bad_data, write_out)

        return cell_manifest_save_path
