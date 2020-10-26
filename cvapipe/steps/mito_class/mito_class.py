#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from aics_dask_utils import DistributedHandler
from datastep import Step, log_run_params

from image_classifier_3d.proj_tester import ProjectTester
from ..prep_analysis_single_cell_ds import PrepAnalysisSingleCellDs

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class MitoClass(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [PrepAnalysisSingleCellDs],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(
            direct_upstream_tasks=direct_upstream_tasks,
            filepath_columns=["crop_raw", "crop_seg"],
            config=config,
        )

    @staticmethod
    def _finalize_cell_annotation(row_index: int, row: pd.Series) -> List:
        """
        Run a pure function.

        Parameters
        ----------
        row_index: int
            the row index
        row: pd.Series
            the row of the cell to be checked

        Return
        ----------
        CellId: int
            ID of the cell
        annotation: int
            one of the following options,
            -1 - N/A
            1 - interphase
            2-
        complete_flag: bool
            if the cell is complete (only valid for M6M7 cells)
        outlier_flag: bool
            if the cell is very likely to be an outlier
        """

        # Keys in Labkey Schema:
        # http://aics.corp.alleninstitute.org/labkey/query/AICS/executeQuery.view?schemaName=processing&query.queryName=MitoticState
        # MitoticStateId = 1 (M6/M7), 2 (M0), 3 (M1/M2), 4 (M3), 5 (M4/M5)

        # Parameters
        uncertain_cutoff = 2.0

        # uncertain cells will be labeled as an outlier
        if np.max(row.pred) < uncertain_cutoff:
            return [row.CellId, -1, True, "Outlier", True]

        # check if it is a bad cell (dead, or blob, or wrong seg)
        if row.pred_label > 5:  # 6 or 7 or 8
            return [row.CellId, -1, True, "Outlier", True]

        # return the label
        if row.pair > 0:  # M7 pair
            return [row.CellId, 1, True, "M6M7_complete", False]
        elif row.pred_label == 0:
            return [row.CellId, 2, True, "M0", False]  # M0
        elif row.pred_label == 1:
            return [row.CellId, 3, True, "M1M2", False]  # M1/M2
        elif row.pred_label == 2:
            return [row.CellId, 4, True, "M3", False]  # M3
        elif row.pred_label == 3:
            return [row.CellId, 5, True, "M4M5", False]  # M4/M5
        elif row.pred_label == 4:
            return [row.CellId, 1, True, "M6M7_complete", False]  # M6 early
        elif row.pred_label == 5:
            return [row.CellId, 1, False, "M6M7_single", False]  # M7 single
        else:
            raise ValueError("invalide value")

    @log_run_params
    def run(
        self,
        dataset: Union[str, Path],
        distributed_executor_address: Optional[str] = None,
        debug: bool = False,
        **kwargs
    ) -> List[Path]:
        """
        Run mitotic classifier and bad cell classifier.

        Parameters
        ----------
        dataset: Union[str, Path]
            The dataset to use for running the classifier.

            **Required dataset columns:** *["CellId", "crop_raw",
            "crop_seg"]*

        distributed_executor_address: Optional[str]
            An optional executor address to pass to some computation engine.

        debug: bool
            A debug flag for the developer to use to manipulate how much data runs,
            how it is processed, etc.
            Default: False (Do not debug)

        Returns
        -------
        manifest_save_path: Path
            Path to the produced manifest of single cell dataset
            with predicted class label
        """
        # path to save intermediate prediction results
        pred_path = self.step_local_staging_dir / "predictions"
        pred_path.mkdir(exist_ok=True)
        my_classifier = ProjectTester(save_model_output=False)
        df_pred = my_classifier.run_tester_csv(dataset, pred_path, return_df=True)

        # load single cell dataset
        df_cells = pd.read_csv(dataset)

        # prepare a dataframe for finalizing annotation
        df_pred_merge = pd.merge(df_pred, df_cells[["CellId", "pair"]], on="CellId")
        with DistributedHandler(distributed_executor_address) as handler:
            # Start processing
            results = handler.batched_map(
                self._finalize_cell_annotation,
                # Convert dataframe iterrows into two lists of items to iterate over
                # One list will be row index
                # One list will be the pandas series of every row
                *zip(*list(df_pred_merge.iterrows())),
                batch_size=20,
            )

        final_annotation = []
        for result in results:
            if not result[4]:  # not an outlier
                final_annotation.append(
                    {
                        "CellId": result[0],
                        "MitoticStateId": result[1],
                        "Complete": result[2],
                        "CellStage": result[3],
                    }
                )
        final_annotation = pd.DataFrame(final_annotation)

        # add final annotation (outliers have been removed) back
        df_analysis = pd.merge(df_cells, final_annotation, on="CellId", how="right")

        # check cell neighbor distance and remove the cells have been removed:
        current_cells = list(df_analysis.CellId.unique())
        for row in df_analysis.itertuples():
            table_index = row.Index
            if row.this_cell_nbr_dist_3d is None or len(row.this_cell_nbr_dist_3d) == 0:
                continue

            update_flag = False
            nbr_dist_3d_old = eval(row.this_cell_nbr_dist_3d)
            nbr_dist_3d_new = []
            nbr_dist_2d_old = eval(row.this_cell_nbr_dist_2d)
            nbr_dist_2d_new = []
            nbr_overlap_old = eval(row.this_cell_nbr_overlap_area)
            nbr_overlap_new = []

            for (neigh_id, neigh_dist) in nbr_dist_3d_old:
                if neigh_id in current_cells:
                    nbr_dist_3d_new.append((neigh_id, neigh_dist))
                else:
                    update_flag = True

            # if no update, then no need to loop others
            if update_flag:
                for (neigh_id, neigh_dist) in nbr_dist_2d_old:
                    if neigh_id in current_cells:
                        nbr_dist_2d_new.append((neigh_id, neigh_dist))

                for (neigh_id, neigh_overlap) in nbr_overlap_old:
                    if neigh_id in current_cells:
                        nbr_overlap_new.append((neigh_id, neigh_overlap))

                df_analysis.at[table_index, "this_cell_nbr_complete"] = 0  # 0 = False
                df_analysis.at[table_index, "this_cell_nbr_dist_3d"] = nbr_dist_3d_new
                df_analysis.at[table_index, "this_cell_nbr_dist_2d"] = nbr_dist_2d_new
                df_analysis.at[
                    table_index, "this_cell_nbr_overlap_area"
                ] = nbr_overlap_new

        self.manifest = df_analysis
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        df_analysis.to_csv(manifest_save_path, index=False)

        return manifest_save_path
