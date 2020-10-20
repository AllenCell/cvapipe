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
    setup_the_autoencoder,
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
        gpu_id=0,
        ref_model_kwargs=dict(
            model_dir="/allen/aics/modeling/gregj/results/integrated_cell"
            + "/test_cbvae_3D_avg_inten/2019-11-27-22:27:04",
            parent_dir="/allen/aics/modeling/gregj/results/integrated_cell/",
            suffix="_94544",
        ),
        targ_model_kwargs=dict(
            parent_dir="/allen/aics/modeling/gregj/results/integrated_cell/",
            model_dir="/allen/aics/modeling/gregj/results/integrated_cell"
            + "/test_cbvae_3D_avg_inten/2019-10-22-15:24:09/",
            suffix="_93300",
        ),
        structures_to_gen=[
            "Desmosomes",
            "Microtubules",
            "Golgi",
            "Nucleolus (Dense Fibrillar Component)",
            "Matrix adhesions",
            "Peroxisomes",
            "Nucleolus (Granular Component)",
            "Endosomes",
            "Mitochondria",
            "Nuclear envelope",
            "Actomyosin bundles",
            "Adherens junctions",
            "Plasma membrane",
            "Actin filaments",
            "Tight junctions",
            "Gap junctions",
            "Lysosome",
        ],
        batch_size=16,
        n_pairs_per_structure=64,
        CellId: Optional[int] = None,
        input_csv_loc: Optional[int] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        GPUId: int
            Which GPU to use
        REF_MODEL_KWARGS: dict
            Dictionary of key to load corresponding trained reference model via IC
        TARG_MODEL_KWARGS: dict
            Dictionary of key to load corresponding trained reference model via IC
        STRUCTURES_TO_GEN: list
            List of structures to generate    
        BATCH_SIZE: int
        N_PAIRS_PER_STRUCTURE: int
            Number of structure pairs to generate to get CI estimates
        CellId: int
        input_csv_loc: pathlib.Path
            Path to input csv containing list of CellIds
            Default: "/allen/aics/modeling/caleb"
        + "/dfNearestDistances_CloserThanK_Euclidean_NumDims_32_K_16.csv",

        Returns
        -------
        result: pathlib.Path
            Path to manifest
        """

        # Get CellIds from input_csv (Caleb's csv by default)
        if input_csv_loc:
            df = pd.read_csv(input_csv_loc)
            all_cellids = df["CellId"]

        if CellId:
            all_cellids = [CellId]

        # Get trained autoencoder, dataprovider
        u_class_names, u_classes, dp, ae = setup_the_autoencoder(
            gpu_id, ref_model_kwargs, targ_model_kwargs
        )

        # Generate all structures that the model is trained on
        if not structures_to_gen or structures_to_gen == "All":
            structures_to_gen = list(u_class_names)

        structure_to_gen_ids = [
            np.where(u_class_names == structure)[0].item()
            for structure in structures_to_gen
        ]
        structure_to_gen_ids = [torch.tensor([x]) for x in structure_to_gen_ids]

        corr_pair_inds = ["i", "j"]

        # Make empty dataframes
        df_all_cellids = pd.DataFrame()

        # Loop over all CellIds
        with tqdm(
            total=len(all_cellids)
            * n_pairs_per_structure
            * len(structures_to_gen * len(corr_pair_inds))
        ) as pbar:
            for this_cell_id in all_cellids:

                # Make empty dataframes
                df = pd.DataFrame()

                # Save generated images to images folder
                image_path = "images" + "_CellID_" + str(this_cell_id)
                image_dir = self.step_local_staging_dir / image_path
                image_dir.mkdir(parents=True, exist_ok=True)

                # grab metadata
                cell_metadata = dp.csv_data[dp.csv_data.CellId == this_cell_id].drop(
                    columns=["level_0", "Unnamed: 0", "index"]
                )

                # search for which split this id is in
                splits = {k for k, v in dp.data.items() if this_cell_id in v["CellId"]}
                assert len(splits) == 1
                split = splits.pop()

                # find the index in the split
                index_in_split = np.where(dp.data[split]["CellId"] == this_cell_id)[0]
                assert len(index_in_split) == 1
                index_in_split = index_in_split[0]

                # grab the sampled image
                _, struct_ind, ref_img = dp.get_sample(
                    train_or_test=split, inds=[index_in_split]
                )

                # move ref image to gpu
                ref = ref_img.cuda()

                for structure, struct_ind in zip(
                    structures_to_gen, structure_to_gen_ids
                ):

                    pbar.set_description(
                        f"Processing {structure} in CellID {this_cell_id}"
                    )

                    df_tmp = cell_metadata[["CellId"]].copy()
                    df_tmp["OriginalTaggedStructure"] = cell_metadata[
                        "Structure"
                    ].item()

                    for ij in corr_pair_inds:
                        df_tmp[f"GeneratedStructureName_{ij}"] = structure
                        df_tmp[f"GeneratedStructureInstance_{ij}"] = -1
                        df_tmp[f"GeneratedStructuePath_{ij}"] = ""
                    df_tmp = pd.concat([df_tmp] * n_pairs_per_structure).reset_index(
                        drop=True
                    )

                    for batch in chunks(range(n_pairs_per_structure), batch_size):

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
                                    image_dir / f"generated_gfp_image_struct_"
                                    f"{struct_safe_name}_instance_{b}_{ij}.ome.tiff"
                                )
                                im_write(im_tensor, img_path)

                                df_tmp.at[b, f"GeneratedStructureInstance_{ij}"] = b
                                df_tmp.at[b, f"GeneratedStructuePath_{ij}"] = img_path

                                pbar.update(1)

                    df = df.append(df_tmp)

                # merge df with metadata for cell
                df = df.reset_index(drop=True)
                df_out = cell_metadata.merge(df)
                this_manifest_save_path = image_dir / "manifest.csv"
                df_out.to_csv(this_manifest_save_path)

                df_all_cellids = df_all_cellids.append(df_out)

        df_all_cellids.reset_index(inplace=True)
        # save out manifest
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        df_all_cellids.to_csv(manifest_save_path)

        return manifest_save_path
