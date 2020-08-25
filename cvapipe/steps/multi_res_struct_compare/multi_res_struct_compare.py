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
    make_pairs_df,
    clean_up_results,
    make_plot,
)

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class MultiResStructCompare(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
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
        px_size=0.29,
        image_dims_crop_size=(64, 160, 96),
        par_dir=Path("/allen/aics/modeling/ritvik/projects/actk/"),
        input_csv_loc=Path("local_staging/singlecellimages/manifest.csv"),
        distributed_executor_address: Optional[str] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        structs: List[str]
            Which tagged structures to include in the analysis.
            Default: [
                "Endoplasmic reticulum",
                "Desmosomes",
                "Mitochondria",
                "Golgi",
                "Microtubules",
                "Nuclear envelope",
                "Nucleolus (Dense Fibrillar Component)",
                "Nucleolus (Granular Component)",
            ]
        N_pairs_per_struct: int
            How many pairs of GFP instances per tagged structure to sample
            from the bigger input dataset.
            Default: 100
        mdata_cols: List[str]
            Which columns from the input dataset to include n the output as metadata
            Default: [
                "StructureShortName",
                "FOVId",
                "CellIndex",
                "CellId",
                "StandardizedFOVPath",
                "CellImage3DPath",
            ]
        px_size: float
            How big are the (cubic) input pixels in micrometers
            Default: 0.29
        image_dims_crop_size: Tuple[int]
            How to crop the input images before the resizing pyamid begins
        par_dir: pathlib.Path
            Parent directory of the input csv, since it's not yet a step.
            Default: Path("/allen/aics/modeling/jacksonb/projects/actk/")
        input_csv_loc: pathlib.Path
            Path to input csv, relative to par_dir
            Default: Path("local_staging/singlecellimages/manifest.csv")

        Returns
        -------
        result: pathlib.Path
            Path to manifest
        """

        # Adding hidden attributes to use in compute distance metric function
        self._par_dir = par_dir
        self._input_csv_loc = input_csv_loc
        self._px_size = px_size

        # Read dataset
        df = pd.read_csv(par_dir / input_csv_loc)

        # subset down to only N_pairs_per_struct
        dataset = make_pairs_df(
            df,
            structs=structs,
            N_pairs_per_struct=N_pairs_per_struct,
            mdata_cols=mdata_cols,
        )

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
                [par_dir for i in range(len(dataset))],
            )

            distance_metric_futures.append(distance_metric_future)
            result = handler.gather(distance_metric_future)
            distance_metric_results.append(result)

        # Assemble final dataframe
        df_final = clean_up_results(distance_metric_results)

        # make a manifest
        self.manifest = pd.DataFrame(columns=["Description", "path"])

        # where to save outputs
        pairwise_loc = (
            self.step_local_staging_dir
            / "pairwise_metrics"
            / "multires_pairwise_similarity.csv"
        )
        pairwise_loc.mkdir(parents=True, exist_ok=True)
        plot_loc = (
            self.step_local_staging_dir
            / "pairwise_plots"
            / "multi_resolution_image_correlation.png"
        )
        plot_loc.mkdir(parents=True, exist_ok=True)

        # save pairwise dataframe to csv
        df_final.to_csv(pairwise_loc, index=False)
        self.manifest = self.manifest.append(
            {"Description": "raw similarity scores", "path": pairwise_loc},
            ignore_index=True,
        )

        # make a plot
        make_plot(data=df_final, save_loc=plot_loc)
        self.manifest = self.manifest.append(
            {"Description": "plot of similarity vs resolution", "path": plot_loc},
            ignore_index=True,
        )

        # save out manifest
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path)

        return manifest_save_path
