#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from aics_dask_utils import DistributedHandler

from datastep import Step, log_run_params

from .utils import (
    compute_distance_metric,
    clean_up_results,
    make_cross_corr_dataframe,
)

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class MultiResCrossCorr(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        mdata_cols=[
            "CellId",
            "CellIndex",
            "FOVId",
            "save_dir",
            "save_reg_path",
            "StructureDisplayName",
            "GeneratedStructureName_i",
            "GeneratedStructureName_j",
            "GeneratedStructureInstance_i",
            "GeneratedStructureInstance_j",
            "GeneratedStructuePath_i",
            "GeneratedStructuePath_j",
        ],
        px_size=0.29,
        image_dims_crop_size=(64, 160, 96),
        input_csv_loc=Path(
            "/allen/aics/modeling/ritvik/projects/cvapipe/local_staging/"
            "/generategfpinstantiations_tmp/images_CellID_987/manifest.csv"
        ),
        distributed_executor_address: Optional[str] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        mdata_cols: List[str]
            Which columns from the input dataset to include in the output as metadata
            Default: [
                "CellId",
                "CellIndex",
                "FOVId",
                "save_dir",
                "save_reg_path",
                "StructureDisplayName",
                "GeneratedStructureName_i",
                "GeneratedStructureName_j",
                "GeneratedStructureInstance_i",
                "GeneratedStructureInstance_j",
                "GeneratedStructuePath_i",
                "GeneratedStructuePath_j",
            ]
        px_size: float
            How big are the (cubic) input pixels in micrometers
            Default: 0.29
        image_dims_crop_size: Tuple[int]
            How to crop the input images before the resizing pyamid begins
        input_csv_loc: pathlib.Path
            Path to input csv
            Default: Path(
                "/allen/aics/modeling/ritvik/projects/cvapipe/local_staging/"
                "/generategfpinstantiations_tmp/images_CellID_987/manifest.csv"
            )

        Returns
        -------
        result: pathlib.Path
            Path to manifest
        """

        # Adding hidden attributes to use in compute distance metric function
        self._input_csv_loc = input_csv_loc
        self._px_size = px_size

        # Read dataset
        df = pd.read_csv(self._input_csv_loc)
        dataset = df[mdata_cols]

        # Make cross correlation dataset
        dataset = make_cross_corr_dataframe(dataset)

        # Empty futures list
        distance_metric_futures = []
        distance_metric_results = []

        # Process each row
        with DistributedHandler(distributed_executor_address) as handler:
            # Start processing
            distance_metric_future = handler.client.map(
                compute_distance_metric,
                [row for i, row in dataset.iterrows()],
                [mdata_cols for i in range(len(dataset))],
                [px_size for i in range(len(dataset))],
                [image_dims_crop_size for i in range(len(dataset))],
            )

            distance_metric_futures.append(distance_metric_future)
            result = handler.gather(distance_metric_future)
            distance_metric_results.append(result)

        # Assemble final dataframe
        df_final = clean_up_results(distance_metric_results)

        # make a manifest
        self.manifest = pd.DataFrame(columns=["Description", "path"])

        # where to save outputs
        pairwise_dir = self.step_local_staging_dir / "pairwise_metrics"
        pairwise_dir.mkdir(parents=True, exist_ok=True)
        pairwise_loc = pairwise_dir / "multires_pairwise_similarity.csv"

        # save pairwise dataframe to csv
        df_final.to_csv(pairwise_loc, index=False)
        self.manifest = self.manifest.append(
            {"Description": "raw similarity scores", "path": pairwise_loc},
            ignore_index=True,
        )

        # save out manifest
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path)

        return manifest_save_path
