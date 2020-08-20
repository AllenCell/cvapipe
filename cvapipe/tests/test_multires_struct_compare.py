#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
import numpy as np

from cvapipe.steps import MultiResStructCompare

EXPECTED_COLUMNS = [
    "CellId_i",
    "CellId_j",
    "Pearson Correlation",
    "Resolution (micrometers)",
]


def test_run_2_structures(data_dir):

    multiresstructcompare = MultiResStructCompare()

    output = multiresstructcompare.run(
        structs=["Actin filaments", "Mitochondria"],
        N_cells_per_struct=2,
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
        input_csv_loc=Path("local_staging_2/singlecellimages/manifest.csv"),
    )

    fig_plus_data_manifest = pd.read_csv(output)
    similarity_score_manifest = pd.read_csv(fig_plus_data_manifest["path"][0])

    # Check that no 2 cells are correlated against each other
    assert np.all(
        similarity_score_manifest["CellId_i"] != similarity_score_manifest["CellId_j"]
    )

    # Check expected columns
    assert all(
        expected_col in similarity_score_manifest.columns
        for expected_col in EXPECTED_COLUMNS
    )
