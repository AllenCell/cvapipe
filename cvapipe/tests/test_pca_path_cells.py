#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd

from cvapipe.steps import PcaPathCells


def test_pca_path_cells(
    N_input_cells=1000,
    pcs=[f"DNA_MEM_PC{i}" for i in range(1, 8 + 1)],
    path=np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]),
    dist_cols=[f"DNA_MEM_PC{i}" for i in range(1, 8 + 1)],
    metric="euclidean",
    id_col="id",
    N_cells=5,
):

    df = pd.DataFrame(
        {
            **{"id": range(N_input_cells)},
            **{
                f"DNA_MEM_PC{i}": np.random.randn(N_input_cells)
                for i in range(1, 8 + 1)
            },
        }
    )
    pca_csv_loc = "example_pcas.csv"
    df.to_csv(pca_csv_loc)

    pcapathcells = PcaPathCells()

    out = pcapathcells.run(
        pca_csv_loc=Path(pca_csv_loc),
        pcs=pcs,
        path=path,
        dist_cols=dist_cols,
        metric=metric,
        id_col=id_col,
        N_cells=N_cells,
    )

    out_manifest = pd.read_csv(out)
    assert len(out_manifest) == len(pcs)

    for i, row in out_manifest.iterrows():
        df_pc = pd.read_csv(row.dataframe_path)
        assert len(df_pc) == len(path) * N_cells
        assert (
            list(df_pc.columns)
            == [id_col, "location", f"{metric} distance to location"] + dist_cols
        )
