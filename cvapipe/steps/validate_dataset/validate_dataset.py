#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Union

import pandas as pd
from datastep import Step, log_run_params

from ...constants import DatasetFields
from ...utils import dataset_utils

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class ValidateDataset(Step):
    """
    A simple step to validate a dataset provided to the workflow.
    """

    DATASET_DESERIALIZERS = {
        ".parquet": pd.read_parquet,
        ".csv": pd.read_csv,
    }

    def __init__(self):
        super().__init__(
            step_name="Raw",  # Manual override to "Raw" for dataset publishing
            filepath_columns=[
                DatasetFields.MembraneContourReadPath,
                DatasetFields.MembraneSegmentationReadPath,
                DatasetFields.NucleusContourReadPath,
                DatasetFields.NucleusSegmentationReadPath,
                DatasetFields.SourceReadPath,
                DatasetFields.StructureContourReadPath,
                DatasetFields.StructureSegmentationReadPath,
            ],
            metadata_columns=[
                DatasetFields.FOVId,
                DatasetFields.CellLine,
                DatasetFields.Gene,
                DatasetFields.PlateId,
                DatasetFields.WellId,
                DatasetFields.ProteinDisplayName,
                DatasetFields.ColonyPosition,
                DatasetFields.GoodCellIndicies,
            ],
        )

    @log_run_params
    def run(
        self, raw_dataset: Union[str, Path] = "aics_p4_data.parquet", **kwargs,
    ) -> Path:
        """
        Validate that the provided dataset contains all the required fields.

        Parameters
        ----------
        raw_dataset: Union[str, Path]
            A path to a local dataset to be used for all downstream processing.
            The dataset will be checked to ensure that all required columns are present.

            Defaults to using the the internally accessible dataset created by
            `scripts/create_aics_dataset.py`: "aics_p4_data.parquet".

            Supported formats: `csv`, `parquet`

        Returns
        -------
        manifest_path: Path
            The local storage path to the manifest to be used for further downstream
            processing and analysis.

        Notes
        -----
        A reminder: The dataset will not be pushed up to quilt during this function run.
        Data is only pushed from local storage to quilt when you run,
        `cvapipe raw push` or `cvapipe all push` this simply gets you and/or validates
        the data you will be working with downstream.
        """
        # Handle dataset provided as string or path
        if isinstance(raw_dataset, (str, Path)):
            dataset = Path(raw_dataset).expanduser().resolve(strict=True)

        # Read dataset
        if dataset.suffix in ValidateDataset.DATASET_DESERIALIZERS:
            dataset = ValidateDataset.DATASET_DESERIALIZERS.get(dataset.suffix)(dataset)
        else:
            raise TypeError(
                f"The provided dataset file is of an unsupported file format. "
                f"Provided: {dataset.suffix}, "
                f"Supported: {list(ValidateDataset.DATASET_DESERIALIZERS.keys())}"
            )

        # Check the dataset for the required columns
        dataset_utils.check_required_fields(
            dataset=dataset,
            required_fields=[*self.filepath_columns, *self.metadata_columns],
        )

        # Save manifest to CSV
        manifest_save_path = self.step_local_staging_dir / "manifest.parquet"
        dataset.to_parquet(manifest_save_path)

        return manifest_save_path
