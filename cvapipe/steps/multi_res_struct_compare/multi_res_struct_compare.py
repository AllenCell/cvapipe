#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from aics_dask_utils import DistributedHandler
from dask.distributed import performance_report

from aicsimageio import AICSImage

from datastep import Step, log_run_params

from .utils import pyramid_correlation, draw_pairs

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
        row: pd.Series,
        mdata_cols: List[str],
    ) -> Union[pd.DataFrame, None]:

        log.info(
            "Beginning pairwise metric computation"
            f" between cells {row.CellId_i}"
            f" and {row.CellId_j}"
        )

        # i and j are abused as variables and str names
        inds = {
            "i": int(row.CellId_i),
            "j": int(row.CellId_j)
        }

        # get data for cells i and j
        gfp_i = masked_gfp_store[row.CellId_i]
        gfp_j = masked_gfp_store[row.CellId_j]
        
        # multi-res comparison
        pyr_corrs = pyramid_correlation(gfp_i, gfp_j, func=np.mean)
    
        # comparison when one input is permuted (as baseline correlation)
        gfp_i_shuf = gfp_i.copy().flatten()
        np.random.shuffle(gfp_i_shuf)
        gfp_i_shuf = gfp_i_shuf.reshape(gfp_i.shape)
        pyr_corrs_permuted = pyramid_correlation(gfp_i_shuf, gfp_j, func=np.mean)
    
        # grab correlations at each res in a df
        df_tmp_corrs = pd.DataFrame()
        for k,v in sorted(pyr_corrs.items()):
            tmp_stat_dict = {
                "Resolution (micrometers)":px_size*k,
                "Pearson Correlation":v,
                "Pearson Correlation permuted":pyr_corrs_permuted[k]
            }
            df_tmp_corrs = df_tmp_corrs.append(tmp_stat_dict, ignore_index=True)
        
        # label stats with cell ids
        df_tmp_corrs["CellId_i"] = row.CellId_i
        df_tmp_corrs["CellId_j"] = row.CellId_j

        # and append row metadata
        df_row_tmp = row.to_frame().T
        df_row_tmp = df_row_tmp.merge(df_tmp_corrs)

        log.info(
            f"Completed pairwise metric between cells {row_i.CellId}"
            f" and {row_j.CellId}"
        )

        return df_row_tmp

    def loop_distance_metric(
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
                # Save Dask report every time row index i is 1
                # This is arbitrary, just want to see the performance
                # once per loop
                if row_index_i == 1:
                    with performance_report(filename="dask-report.html"):
                        df_row_tmp = self.compute_distance_metric(
                            row_index_i, row_i, row_index_j, row_j, mdata_cols,
                        )
                else:
                    df_row_tmp = self.compute_distance_metric(
                        row_index_i, row_i, row_index_j, row_j, mdata_cols,
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
        image_dims_crop_size = (64, 160, 96),
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
            How many pairs of GFP instances per tagged structure to sample from the bigger input dataset.
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

        # Handle dataset provided as string or path
        if isinstance(par_dir / input_csv_loc, (str, Path)):
            df = Path(par_dir / input_csv_loc).expanduser().resolve(strict=True)

            # Read dataset
            df = pd.read_csv(par_dir / input_csv_loc)

            
        # subset down to only N_pairs_per_struct
        dataset = pd.DataFrame()

        for struct in structs:
            try:
                df_struct = df[df.StructureShortName == struct]
                pair_inds = draw_pairs(df_struct.index, n_pairs=N_pairs_per_struct)
                inds_i = np.array(list(pair_inds))[:,0]
                inds_j = np.array(list(pair_inds))[:,1]
                df_struct_i = df_struct.loc[inds_i]
                df_struct_j = df_struct.loc[inds_j]
                for (i,j) in pair_inds:
                    row_i = df_struct.loc[i]
                    row_j = df_struct.loc[j]
                    row = row_i[mdata_cols].add_suffix("_i").append(row_j[mdata_cols].add_suffix("_j"))
                    dataset = dataset.append(row, ignore_index=True)
            except Exception as e:
                log.info(
                    f"Not enough {struct} rows to get {N_pairs_per_struct} samples"
                    f" Error {e}."
                )
                break
        assert np.all(dataset["CellId_i"] != dataset["CellId_j"])
        dataset = dataset.reset_index(drop=True)

        # Empty futures list
        distance_metric_futures = []
        distance_metric_results = []

        # Process each row
        with DistributedHandler(distributed_executor_address) as handler:
            for this_row_index, this_row in dataset.iterrows():
                # Start processing
                distance_metric_future = handler.client.map(
                    self.loop_distance_metric,
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
        df_final["Pearson Correlation gain over random"] = df_final["Pearson Correlation"] - df_final["Pearson Correlation permuted"]

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
        sns.set(style="ticks", rc={"lines.linewidth": 1.0})
        fig = plt.figure(figsize=(10,7))

        ax = sns.pointplot(
            x="Resolution (micrometers)",
            y="Pearson Correlation gain over random",
            hue="StructureShortName",
            data=df_pairwise_corrs,
            ci=95,
            capsize=.2,
            palette="Set2",
        )
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(0.05, 0.95),
            ncol=1,
            frameon=False,
        )
        sns.despine(
            offset=0,
            trim=True,
        );

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
