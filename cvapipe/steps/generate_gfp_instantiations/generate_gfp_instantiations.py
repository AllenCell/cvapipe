#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

from tqdm import tqdm

import pandas as pd
import numpy as np
import torch

import sys

# Insert path to an IC module
sys.path.insert(1, "/allen/aics/modeling/ritvik/projects/pytorch_integrated_cell/")

from integrated_cell import utils

# from aics_dask_utils import DistributedHandler

from datastep import Step, log_run_params

# from aicsimageio.writers import OmeTiffWriter

from .gen_gfp_utils import (
    im_write,
    chunks,
    SetupTheAutoencoder,
)

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class GenerateGFPInstantiations(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        GPUId=4,
        REF_MODEL_KWARGS=dict(
            model_dir="/allen/aics/modeling/gregj/results/integrated_cell"
            + "/test_cbvae_3D_avg_inten/2019-11-27-22:27:04",
            parent_dir="/allen/aics/modeling/gregj/results/integrated_cell/",
            suffix="_94544",
        ),
        TARG_MODEL_KWARGS=dict(
            parent_dir="/allen/aics/modeling/gregj/results/integrated_cell/",
            model_dir="/allen/aics/modeling/gregj/results/integrated_cell"
            + "/test_cbvae_3D_avg_inten/2019-10-22-15:24:09/",
            suffix="_93300",
        ),
        STRUCTURES_TO_GEN=[
            "Desmosomes",
            "Endoplasmic reticulum",
            "Golgi",
            "Microtubules",
            "Mitochondria",
            "Nuclear envelope",
            "Nucleolus (Dense Fibrillar Component)",
            "Nucleolus (Granular Component)",
        ],
        BATCH_SIZE=16,
        N_PAIRS_PER_STRUCTURE=64,
        input_csv_loc="/allen/aics/modeling/caleb"
        + "/dfNearestDistances_CloserThanK_Euclidean_NumDims_32_K_16.csv",
        **kwargs,
    ):
        """
        Parameters
        ----------
        mdata_cols: List[str]
            Which columns from the input dataset to include in the output as metadata
            Default: [
                "CellId",
                "CellIndex",
                "FOVId",
                "save_dir",
                "save_reg_path",
                "StructureDisplayName",
                "GeneratedStructureName_i",
                "GeneratedStructureName_j",
                "GeneratedStructureInstance_i",
                "GeneratedStructureInstance_j",
                "GeneratedStructuePath_i",
                "GeneratedStructuePath_j",
            ]
        px_size: float
            How big are the (cubic) input pixels in micrometers
            Default: 0.29
        image_dims_crop_size: Tuple[int]
            How to crop the input images before the resizing pyamid begins
        input_csv_loc: pathlib.Path
            Path to input csv
            Default: Path(
                "/allen/aics/modeling/rorydm/results/multiscale_structure_similarity"\
                "/generated_gfp_images/generated_structures.csv"
            )

        Returns
        -------
        result: pathlib.Path
            Path to manifest
        """

        # Get CellIds from input_csv (Caleb's csv by default)
        df_cellids = pd.read_csv(input_csv_loc)
        all_cellids = list(df_cellids.CellId)

        # Get trained autoencoder, dataprovider
        u_class_names, u_classes, dp, ae = SetupTheAutoencoder(
            GPUId, REF_MODEL_KWARGS, TARG_MODEL_KWARGS
        )

        # Save generated images to images folder
        image_dir = self.step_local_staging_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        # Generate all structures that the model is trained on
        if not STRUCTURES_TO_GEN or STRUCTURES_TO_GEN == "All":
            STRUCTURES_TO_GEN = list(u_class_names)

        structure_to_gen_ids = [
            np.where(u_class_names == structure)[0].item()
            for structure in STRUCTURES_TO_GEN
        ]
        structure_to_gen_ids = [torch.tensor([x]) for x in structure_to_gen_ids]

        # Make empty dataframes
        df = pd.DataFrame()
        all_cell_metadata = pd.DataFrame()

        corr_pair_inds = ["i", "j"]

        # Loop over all CellIds
        with tqdm(
            total=len(all_cellids)
            * N_PAIRS_PER_STRUCTURE
            * len(STRUCTURES_TO_GEN * len(corr_pair_inds))
        ) as pbar:
            for ThisCellId in all_cellids:

                # pick this based on pca location or whatever
                MY_CELL_ID = ThisCellId

                # grap metadata
                cell_metadata = dp.csv_data[dp.csv_data.CellId == MY_CELL_ID].drop(
                    columns=["level_0", "Unnamed: 0"]
                )

                # find dp index from ID
                cell_index = cell_metadata.index.item()
                # cell_index = cell_metadata['CellId'].item()

                # search for which split this id is in
                for k, v in dp.data.items():
                    if cell_index in v["inds"]:
                        split = k

                # grab the sampled image
                gfp_img, struct_ind, ref_img = dp.get_sample(
                    train_or_test=split, inds=[cell_index]
                )

                # move ref image to gpu
                ref = ref_img.cuda()

                for structure, struct_ind in zip(
                    STRUCTURES_TO_GEN, structure_to_gen_ids
                ):

                    pbar.set_description(f"Processing {structure}")

                    df_tmp = cell_metadata[["CellId"]].copy()
                    df_tmp["OriginalTaggedStructure"] = cell_metadata[
                        "Structure"
                    ].item()

                    for ij in corr_pair_inds:
                        df_tmp[f"GeneratedStructureName_{ij}"] = structure
                        df_tmp[f"GeneratedStructureInstance_{ij}"] = -1
                        df_tmp[f"GeneratedStructuePath_{ij}"] = ""
                    df_tmp = pd.concat([df_tmp] * N_PAIRS_PER_STRUCTURE).reset_index(
                        drop=True
                    )

                    for batch in chunks(range(N_PAIRS_PER_STRUCTURE), BATCH_SIZE):

                        # one hot structure labels to generate,same for whole batch
                        labels_gen_batch = (
                            utils.index_to_onehot(struct_ind, len(u_classes))
                            .repeat(len(batch), 1)
                            .cuda()
                        )

                        # repeat reference structure over batch
                        ref_batch = ref.repeat([len(batch), 1, 1, 1, 1])

                        # we want two images per row, image i and image j,
                        #  so no hidden corr in aggreagate metrics
                        for ij in corr_pair_inds:

                            # generate our gfp samples
                            target_gen_batch, _ = ae(
                                target=None, ref=ref_batch, labels=labels_gen_batch
                            )

                            # save images
                            for b, im_tensor in zip(batch, target_gen_batch):
                                struct_safe_name = structure.replace(" ", "_").lower()
                                img_path = (
                                    image_dir
                                    / f"generated_gfp_image_struct_{struct_safe_name}"
                                    f"_instance_{b}_{ij}.ome.tiff"
                                )
                                im_write(im_tensor, img_path)

                                df_tmp.at[b, f"GeneratedStructureInstance_{ij}"] = b
                                df_tmp.at[b, f"GeneratedStructuePath_{ij}"] = img_path

                                pbar.update(1)

                    df = df.append(df_tmp)
                    all_cell_metadata = all_cell_metadata.append(cell_metadata)

            # merge df with metadata for cell
            df = df.reset_index(drop=True)
            df_out = all_cell_metadata.merge(df)

        # save df
        self.manifest = df_out
        # save out manifest
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path)

        return manifest_save_path
