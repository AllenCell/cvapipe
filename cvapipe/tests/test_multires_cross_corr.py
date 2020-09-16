#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd

from cvapipe.steps import MultiResCrossCorr

EXPECTED_COLUMNS = [
    "GeneratedStructureName_i_1",
    "GeneratedStructureName_i_2",
    "Pearson Correlation",
    "Resolution (micrometers)",
]


def test_run(data_dir):

    multirescrosscorr = MultiResCrossCorr()

    output = multirescrosscorr.run(
        input_csv_loc=Path(
            "/allen/aics/modeling/ritvik/projects/cvapipe/"
            + "cvapipe/tests/data/example_dataset_multi_res_struct_compare.csv"
        ),
    )

    fig_plus_data_manifest = pd.read_csv(output)
    similarity_score_manifest = pd.read_csv(fig_plus_data_manifest["path"][0])

    # Check expected columns
    assert all(
        expected_col in similarity_score_manifest.columns
        for expected_col in EXPECTED_COLUMNS
    )
