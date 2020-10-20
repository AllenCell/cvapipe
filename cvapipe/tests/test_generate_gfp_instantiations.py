#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

from cvapipe.steps import GenerateGFPInstantiations

from cvapipe.steps.multi_res_struct_compare.constants import DatasetFieldsIC


def test_run(data_dir):

    generategfpinstantations = GenerateGFPInstantiations()

    output = generategfpinstantations.run(
        structures_to_gen=["Desmosomes", "Microtubules"],
        batch_size=16,
        n_pairs_per_structure=2,
        CellId=9445,
    )

    manifest = pd.read_csv(output)

    # Check dataset fields
    assert all(
        expected_col in manifest.columns
        for expected_col in [
            DatasetFieldsIC.CellId,
            DatasetFieldsIC.StructureName1,
            DatasetFieldsIC.StructureName2,
            DatasetFieldsIC.SourceReadPath1,
            DatasetFieldsIC.SourceReadPath2,
            DatasetFieldsIC.CellIndex,
            DatasetFieldsIC.FOVId,
            DatasetFieldsIC.SaveDir,
            DatasetFieldsIC.SaveRegPath,
            DatasetFieldsIC.GeneratedStructureInstance_i,
            DatasetFieldsIC.GeneratedStructureInstance_j,
        ]
    )
