#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

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

    @log_run_params
    def run(
        self,
        structs=[
            "Endoplasmic reticulum",
            "Desmosomes",
            "Mitochondria",
            "Golgi",
            "Microtubules",
            "Nuclear envelope",
            "Nucleolus (Dense Fibrillar Component)",
            "Nucleolus (Granular Component)",
        ],
        N_cells_per_struct=100,
        mdata_cols=[
            "StructureShortName",
            "FOVId",
            "CellIndex",
            "CellId",
            "StandardizedFOVPath",
            "CellImage3DPath",
        ],
        px_size=0.29,
        par_dir=Path("/allen/aics/modeling/jacksonb/projects/actk/"),
        input_csv_loc=Path("local_staging/singlecellimages/manifest.csv"),
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

        # grab the inputs since they're not in a step yet
        new_df_path = par_dir / input_csv_loc
        df = pd.read_csv(new_df_path)

        # subset down to only N_cells_per_struct per structure
        df_short_sample = pd.DataFrame()
        for struct in structs:
            df_short_sample = df_short_sample.append(
                df[df.StructureShortName == struct].sample(N_cells_per_struct)
            )
        df_short_sample = df_short_sample.reset_index(drop=True)

        # make df of cell pairs + metadata
        df_pairwise = pd.DataFrame()
        for struct in structs:
            df_struct = df_short_sample[df_short_sample.StructureShortName == struct]
            for i, row_i in df_struct.iterrows():
                for j, row_j in df_struct.iterrows():
                    if j > i:
                        row = (
                            row_i[mdata_cols]
                            .add_suffix("_i")
                            .append(row_j[mdata_cols].add_suffix("_j"))
                        )
                        df_pairwise = df_pairwise.append(row, ignore_index=True)
        assert np.all(df_pairwise["CellId_i"] != df_pairwise["CellId_j"])

        # go through each pair and measure similarity at each resolution
        df_pairwise_corrs = pd.DataFrame()
        for k, row in tqdm(df_pairwise.iterrows(), total=len(df_pairwise)):

            # load first cell and check
            image_i = AICSImage(par_dir / row.CellImage3DPath_i)
            assert image_i.get_channel_names()[4] == "structure"
            image_i_gfp_3d = image_i.get_image_data("ZYX", S=0, T=0, C=4)
            image_i_gfp_3d_bool = image_i_gfp_3d > 0

            # load second cell and check
            image_j = AICSImage(par_dir / row.CellImage3DPath_j)
            assert image_j.get_channel_names()[4] == "structure"
            image_j_gfp_3d = image_j.get_image_data("ZYX", S=0, T=0, C=4)
            image_j_gfp_3d_bool = image_j_gfp_3d > 0

            # correlations at each resolution of image pyramid
            pyr_corrs = pyramid_correlation(image_i_gfp_3d_bool, image_j_gfp_3d_bool)

            # tmp df to save stats for each resolution of these cells
            df_tmp_corrs = pd.DataFrame()
            for k, v in sorted(pyr_corrs.items()):
                tmp_stat_dict = {
                    "Resolution (micrometers)": px_size * k,
                    "Pearson Correlation": v,
                }
                df_tmp_corrs = df_tmp_corrs.append(tmp_stat_dict, ignore_index=True)
            df_tmp_corrs["CellId_i"] = row.CellId_i
            df_tmp_corrs["CellId_j"] = row.CellId_j

            # merge stats df to row
            df_row_tmp = row.to_frame().T
            df_row_tmp = df_row_tmp.merge(df_tmp_corrs)

            # append stats for these two cells to main output
            df_pairwise_corrs = df_pairwise_corrs.append(df_row_tmp)

        # fix up output df
        df_pairwise_corrs = df_pairwise_corrs.reset_index(drop=True)
        df_pairwise_corrs.shape
        df_pairwise_corrs = df_pairwise_corrs.rename(
            columns={"StructureShortName_i": "StructureShortName"}
        ).drop(columns="StructureShortName_j")

        # make a manifest
        self.manifest = pd.DataFrame(columns=["Description", "path"])

        # where to save outputs
        pairwise_dir = self.step_local_staging_dir / "pairwise_metrics"
        pairwise_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = self.step_local_staging_dir / "pairwise_metrics"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # save df
        df_pairwise_corrs.to_csv(
            pairwise_dir / "multires_pairwise_similarity.csv", index=False
        )
        self.manifest = self.manifest.append(
            {"Description": "raw similarity scores", "path": "foo/bar"},
            ignore_index=True,
        )

        # make a plot
        fig = plt.figure(figsize=(10, 7))
        ax = sns.pointplot(
            x="Resolution (micrometers)",
            y="Pearson Correlation",
            hue="StructureShortName",
            data=df_pairwise_corrs,
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
            "multi_resolution_image_correlation.png",
            format="png",
            dpi=300,
            transparent=True,
        )

        # save out manifest
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path)
        return manifest_save_path
