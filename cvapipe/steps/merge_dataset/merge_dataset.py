#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from datastep import Step, log_run_params

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class MergeDataset(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        dataset_with_annotation: Union[str, Path],
        dataset_from_labkey: Union[str, Path],
        debug: bool = False,
        **kwargs
    ) -> List[Path]:
        """
        Merge the mitotic label with original labkey query.

        Protected Parameters
        --------------------
        debug: bool
            A debug flag for the developer to use to manipulate how much data runs,
            how it is processed, etc.
            Default: False (Do not debug)

        Parameters
        ----------
        dataset_with_annotation: Union[str, Path]
            The dataset containing mitotic annotation. Outliers have been removed.

            **Required dataset columns:** *["CellId", "MitoticStateId",
            "Complete"]*

        dataset_from_labkey: Union[str, Path]
            The dataset originally from labkey query

        Returns
        -------
        manifest_save_path: Path
            Path to the produced manifest of cell table for CFE
        """

        df_labkey = pd.read_parquet(
            Path(dataset_from_labkey).expanduser().resolve(strict=True)
        )

        df_anno = pd.read_csv(
            dataset_with_annotation, usecols=["CellId", "MitoticStateId", "Complete"]
        )

        # merge the two tables, annotation table may be smaller, because
        # outliers have been removed, cells with failed QC have been excluded.
        df_merge = pd.merge(df_labkey, df_anno, on="CellId", how="right")

        self.manifest = df_merge
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        df_merge.to_csv(manifest_save_path, index=False)

        return manifest_save_path
