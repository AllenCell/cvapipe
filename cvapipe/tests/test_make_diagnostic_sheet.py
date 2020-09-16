#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd

from cvapipe.steps import MakeDiagnosticSheet


EXPECTED_COLUMNS = [
    "StructureName",
    "HTMLSavePath",
]


def test_run_2_structures(data_dir):

    makediagnosticsheet = MakeDiagnosticSheet()

    output = makediagnosticsheet.run(
        data_dir / "example_make_diagnostic_sheet.csv",
        structs=["Desmosomes", "Nucleolus (Dense Fibrillar Component)"],
        ncells=2,
    )

    output_manifest = pd.read_csv(output)

    # Run asserts
    # Check expected columns
    assert all(
        expected_col in output_manifest.columns for expected_col in EXPECTED_COLUMNS
    )

    # Check all expected files exist
    for field in [
        "HTMLSavePath",
    ]:
        assert all(Path(f).resolve(strict=True) for f in output_manifest[field])
