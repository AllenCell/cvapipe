#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from aics_dask_utils import DistributedHandler

from aicsimageio import AICSImage

from datastep import Step, log_run_params

from .utils import pyramid_correlation

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

    def compute_distance_metric(
        self,
        row_index_i: int,
        row_i: pd.Series,
        row_index_j: int,
        row_j: pd.Series,
        mdata_cols: List,
    ) -> Union[pd.DataFrame, None]:

        if row_i["StructureShortName"] == row_j["StructureShortName"]:
            if row_index_j > row_index_i:

                assert row_i["CellId"] != row_j["CellId"]

                log.info(
                    "Beginning pairwise metric computation"
                    f" between cells {row_i.CellId}"
                    f" and {row_j.CellId}"
                )

                row = (
                    row_i[mdata_cols]
                    .add_suffix("_i")
                    .append(row_j[mdata_cols].add_suffix("_j"))
                )

                # load first cell and check
                image_i = AICSImage(self._par_dir / row.CellImage3DPath_i)
                assert image_i.get_channel_names()[4] == "structure"
                image_i_gfp_3d = image_i.get_image_data("ZYX", S=0, T=0, C=4)
                image_i_gfp_3d_bool = image_i_gfp_3d > 0

                # load second cell and check
                image_j = AICSImage(self._par_dir / row.CellImage3DPath_j)
                assert image_j.get_channel_names()[4] == "structure"
                image_j_gfp_3d = image_j.get_image_data("ZYX", S=0, T=0, C=4)
                image_j_gfp_3d_bool = image_j_gfp_3d > 0

                # correlations at each resolution of image pyramid
                pyr_corrs = pyramid_correlation(
                    image_i_gfp_3d_bool, image_j_gfp_3d_bool
                )

                # tmp df to save stats for each resolution of these cells
                df_tmp_corrs = pd.DataFrame()
                for k, v in sorted(pyr_corrs.items()):
                    tmp_stat_dict = {
                        "Resolution (micrometers)": self._px_size * k,
                        "Pearson Correlation": v,
                    }
                    df_tmp_corrs = df_tmp_corrs.append(tmp_stat_dict, ignore_index=True)
                df_tmp_corrs["CellId_i"] = row.CellId_i
                df_tmp_corrs["CellId_j"] = row.CellId_j

                # merge stats df to row
                df_row_tmp = row.to_frame().T
                df_row_tmp = df_row_tmp.merge(df_tmp_corrs)

                log.info(
                    f"Completed pairwise metric between cells {row_i.CellId}"
                    f" and {row_j.CellId}"
                )

                return df_row_tmp

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
        N_cells_per_struct=70,
        mdata_cols=[
            "StructureShortName",
            "FOVId",
            "CellIndex",
            "CellId",
            "StandardizedFOVPath",
            "CellImage3DPath",
        ],
        px_size=0.29,
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
        N_cells_per_struct: int
            How many cells per tagged structure to sample from the bigger input dataset.
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

        # Handle dataset provided as string or path
        if isinstance(par_dir / input_csv_loc, (str, Path)):
            df = Path(par_dir / input_csv_loc).expanduser().resolve(strict=True)

            # Read dataset
            df = pd.read_csv(par_dir / input_csv_loc)

        # subset down to only N_cells_per_struct per structure
        dataset = pd.DataFrame()

        for struct in structs:
            try:
                dataset = dataset.append(
                    df[df.StructureShortName == struct].sample(N_cells_per_struct)
                )
            except Exception as e:
                log.info(
                    f"Not enough {struct} rows to get {N_cells_per_struct} samples"
                    f" Error {e}."
                )
                break

        dataset = dataset.reset_index(drop=True)

        # Empty futures list
        distance_metric_futures = []
        distance_metric_results = []

        # Process each row
        with DistributedHandler(distributed_executor_address) as handler:
            for this_row_index, this_row in dataset.iterrows():
                # Start processing
                distance_metric_future = handler.client.map(
                    self.compute_distance_metric,
                    # Convert dataframe iterrows into two lists of items to iterate over
                    # One list will be row index
                    # One list will be the pandas series of every row
                    *zip(*list(dataset.iterrows())),
                    # Keep the other row (row j) constant as we
                    # loop through the dataset (row i)
                    [this_row_index for i in range(len(dataset))],
                    [this_row for i in range(len(dataset))],
                    [mdata_cols for i in range(len(dataset))],
                )

                distance_metric_futures.append(distance_metric_future)

                result = handler.gather(distance_metric_future)
                distance_metric_results.append(result)

            # This seems to be a lot slower and clogs a single core
            #  Collect futures
            # distance_metric_results = [
            #     handler.gather(f) for f in distance_metric_futures
            # ]

        # Assemble final dataframe
        df_final = pd.DataFrame()
        for dataframes in distance_metric_results:
            for corr_dataframe in dataframes:
                # corr_dataframe is None if
                # row_i["StructureShortName"] != row_j["StructureShortName"]
                # or row_index_j > row_index_i
                if corr_dataframe is not None:
                    df_final = df_final.append(corr_dataframe)

        log.info(f"Assembled Parwise metrics dataframe with shape {df_final.shape}")

        # fix up final pairwise dataframe
        df_final = df_final.reset_index(drop=True)
        df_final = df_final.rename(
            columns={"StructureShortName_i": "StructureShortName"}
        ).drop(columns="StructureShortName_j")

        # make a manifest
        self.manifest = pd.DataFrame(columns=["Description", "path"])

        # where to save outputs
        pairwise_dir = self.step_local_staging_dir / "pairwise_metrics"
        pairwise_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = self.step_local_staging_dir / "pairwise_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # save pairwise dataframe to csv
        df_final.to_csv(pairwise_dir / "multires_pairwise_similarity.csv", index=False)
        self.manifest = self.manifest.append(
            {
                "Description": "raw similarity scores",
                "path": pairwise_dir / "multires_pairwise_similarity.csv",
            },
            ignore_index=True,
        )

        # make a plot
        fig = plt.figure(figsize=(10, 7))
        ax = sns.pointplot(
            x="Resolution (micrometers)",
            y="Pearson Correlation",
            hue="StructureShortName",
            data=df_final,
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
            plots_dir / "multi_resolution_image_correlation.png",
            format="png",
            dpi=300,
            transparent=True,
        )
        self.manifest = self.manifest.append(
            {
                "Description": "plot of similarity vs resolution",
                "path": plots_dir / "multi_resolution_image_correlation.png",
            },
            ignore_index=True,
        )

        # save out manifest
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path)

        return manifest_save_path
