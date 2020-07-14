#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest

from cvapipe import exceptions
from cvapipe.steps import ValidateDataset


@pytest.mark.parametrize(
    "raw_dataset",
    [
        "example_dataset_succeeding.parquet",
        pytest.param("example.txt", marks=pytest.mark.raises(exception=TypeError)),
        pytest.param(
            "example_dataset_failing.csv",
            marks=pytest.mark.raises(exception=exceptions.MissingDataError),
        ),
    ],
)
def test_validate_dataset_run(data_dir, tmpdir, raw_dataset):
    # Initialize step
    validate_dataset = ValidateDataset()

    # Construct full path for data
    raw_dataset = data_dir / raw_dataset

    # Run with data
    manifest_path = validate_dataset.run(raw_dataset=raw_dataset)

    assert isinstance(manifest_path, Path)
