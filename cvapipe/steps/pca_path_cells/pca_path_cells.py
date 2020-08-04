#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from datastep import Step, log_run_params
from .utils import scan_pc_for_cells

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class PcaPathCells(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
        filepath_columns=["dataframe_loc"],
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        pca_csv_loc=Path(
            "/allen/aics/assay-dev/MicroscopyOtherData/Viana/forCaleb/variance/05202020_Align-IND_Chirality-OFF/manifest.csv"
        ),
        pcs=[1, 2, 3, 4, 5, 6, 7, 8],
        path=np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]),
        dist_cols=[
            "DNA_MEM_PC1",
            "DNA_MEM_PC2",
            "DNA_MEM_PC3",
            "DNA_MEM_PC4",
            "DNA_MEM_PC5",
            "DNA_MEM_PC6",
            "DNA_MEM_PC7",
            "DNA_MEM_PC8",
        ],
        metric="euclidean",
        id_col="CellId",
        N_cells=3,
        **kwargs,
    ):
        """
        Look through PCA embeddings of cells and find groups of cells closest to PC axes.

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
        pca_csv_loc: pathlib.Path
            Location of csv containing pca embeddings of cells
            Default: Path("/allen/aics/assay-dev/MicroscopyOtherData/Viana/forCaleb/variance/05202020_Align-IND_Chirality-OFF/manifest.csv")
        pcs: List[int]
            Which pcs do we want to trace through
            Default: [1,2,3,4,5,6,7,8]
        path: np.array
            Path containing points along each PC axis where we find nearest cells
            Default: np.array([-2.0, -1.5, -1.0, -0.5,  0.0,  0.5,  1.0,  1.5,  2.0])
        dist_cols: List[str]
            Which columns in the `pca_csv_loc` contribute to distance computations?
            Default: [
                'DNA_MEM_PC1',
                'DNA_MEM_PC2',
                'DNA_MEM_PC3',
                'DNA_MEM_PC4',
                'DNA_MEM_PC5',
                'DNA_MEM_PC6',
                'DNA_MEM_PC7',
                'DNA_MEM_PC8'
            ]
        metric: str
            How do we compute distance? This kwarg is passed to scipy.spatial.distance.cdist
            Default: "euclidean"
        id_col: str
            Which column in `pca_csv_loc` is used for unique cell ids?
            Default: "CellId"
        N_cells: int
            How many nearest cells to each point on `path` are returned?
            Default: 3

        Returns
        -------
        result: self.manifest
        """

        self.manifest = pd.DataFrame(columns=["PC", "dataframe_path"])
        pc_path_dir = (self.step_local_staging_dir / "pc_paths").mkdir(
            parents=True, exist_ok=True
        )

        # TODO change this filepath location to an upstream step
        df_pca = pd.read_csv(pca_csv_loc)

        for pc in pcs:
            df_cells = scan_pc_for_cells(
                df_pca,
                pc=pc,
                path=path,
                dist_cols=dist_cols,
                metric=metric,
                id_col=id_col,
                N_cells=N_cells,
            )

            fpath = pc_path_dir / f"pca_{pc}_path_cells.csv"
            df_cells.to_csv(fpath, index=False)

            self.manifest = self.manifest.append({"PC": pc, "dataframe_path": fpath})

        return self.manifest