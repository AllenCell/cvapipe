import json
import logging
from pathlib import Path
from typing import List, NamedTuple, Optional, Union, Dict

import aicsimageio
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aics_dask_utils import DistributedHandler
from aicsimageio import AICSImage, transforms
from aicsimageio.writers import OmeTiffWriter
import aicsimageprocessing as proc
from datastep import Step, log_run_params
from imageio import imwrite

from .diagnostic_sheet_utils import (
    get_normed_image_array,
    select_and_adjust_segmentation_ceiling,
    crop_raw_channels_with_segmentation,
    get_thumbnail,
    HTMLWriter,
)

# get_slice_highest_intensity,
# pct_normalization_and_8bit,
# minmax_normalization_and_8bit,
# get_bottom_and_top_slices,
# get_slice_range,
# get_cell_center,
# get_top,

plt.style.use("dark_background")

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class ZSliceRanges:
    CETN2 = ([90, 100], "mem")
    TUBA1B = ([88, 88], "mem")
    PXN = ([0, 10], "mem")
    TJP1 = ([88, 88], "mem")
    LMNB1 = ([50, 50], "nuc")
    NUP153 = ([50, 50], "nuc")
    ST6GAL1 = ([88, 88], "mem")
    LAMP1 = ([88, 88], "mem")
    ACTB = ([88, 88], "mem")
    DSP = ([88, 88], "mem")
    FBL = ([50, 50], "nuc")
    NPM1 = ([50, 50], "nuc")
    TOMM20 = ([88, 88], "mem")
    SLC25A17 = ([88, 88], "mem")
    ACTN1 = ([88, 88], "mem")
    GJA1 = ([88, 88], "mem")
    H2B = ([50, 50], "nuc")
    SON = ([50, 50], "nuc")
    SEC61B = ([88, 88], "mem")
    RAB5A = ([88, 88], "mem")
    MYH10 = ([88, 88], "mem")
    AAVS1 = ([88, 88], "mem")
    CTNNB1 = ([88, 88], "mem")
    ATP2A2 = ([88, 88], "mem")
    SMC1A = ([50, 50], "nuc")


class DatasetFields:
    GeneratedStructureName = "GeneratedStructureName_i"
    GeneratedStructureInstance = "GeneratedStructureInstance_i"
    CellImage2DAllProjectionsPath = "CellImage2DAllProjectionsPath"
    CellImage3DPath = "CellImage3DPath"
    CellImage2DYXProjectionPath = "CellImage2DYXProjectionPath"
    DiagnosticSheetPath = "DiagnosticSheetPath"


class DiagnosticSheetResult(NamedTuple):
    cell_id: Union[int, str]
    save_path: Optional[Path] = None


class DiagnosticSheetError(NamedTuple):
    cell_id: Union[int, str]
    error: str


class CellImagesResult(NamedTuple):
    generated_struct_name: str
    gen_struct_inst: int
    path_3d: Path
    path_2d_all_proj: Path
    path_2d_yx_proj: Path


class CellImagesError(NamedTuple):
    cell_id: Union[int, str]
    error: str


###############################################################################


class MakeDiagnosticSheet(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @staticmethod
    def _normalize_images(
        row_index: int,
        row: pd.Series,
        cell_ceiling_adjustment: int,
        bounding_box: np.ndarray,
        current_pixel_sizes=(0.10833333333333332, 0.10833333333333332, 0.29,),
        desired_pixel_sizes=(0.29, 0.29, 0.29),
    ):
        # Don't use dask for image reading
        aicsimageio.use_dask(False)

        # Only do the image normalization for the first image
        # Every row is based on the same source image
        if row.GeneratedStructureInstance_i == 0:

            log.info(f"Starting image normalization for row index {row_index}")

            (normalized_img, channels, pixel_sizes,) = get_normed_image_array(
                raw_image=row.SourceReadPath,
                nucleus_seg_image=row.NucleusSegmentationReadPath,
                membrane_seg_image=row.MembraneSegmentationReadPath,
                dna_channel_index=5,
                membrane_channel_index=1,
                structure_channel_index=3,
                brightfield_channel_index=6,
                nucleus_seg_channel_index=0,
                membrane_seg_channel_index=0,
                current_pixel_sizes=current_pixel_sizes,
                desired_pixel_sizes=desired_pixel_sizes,
            )

            # Select and adjust cell shape ceiling for this cell
            image = select_and_adjust_segmentation_ceiling(
                # Unlike most other operations, we can read in normal "CZYX" dimension
                # order here as all future operations are expecting it
                image=normalized_img,
                cell_index=row.CellIndex,
                cell_ceiling_adjustment=cell_ceiling_adjustment,
            )

            # Perform a rigid registration on the image
            image, _, _ = proc.cell_rigid_registration(
                image,
                # Reorder bounding box as image is currently CYXZ
                bbox_size=bounding_box[[0, 2, 3, 1]],
            )

            # Generate 2d image projections
            # Crop raw channels using segmentations
            image = crop_raw_channels_with_segmentation(image, channels)

            image = transforms.transpose_to_dims(image, "CYXZ", "CZYX")

            log.info(f"Completed image normalization for row index {row_index}")

            return image
        else:
            return None

    @staticmethod
    def _generate_single_cell_images(
        row_index: int,
        row: pd.Series,
        image: np.ndarray,
        projection_method: str,
        projection_channels: str,
        cell_images_3d_dir: Path,
        cell_images_2d_all_proj_dir: Path,
        cell_images_2d_yx_proj_dir: Path,
        overwrite: bool,
        original_structure: bool,
        current_pixel_sizes=(0.10833333333333332, 0.10833333333333332, 0.29,),
        desired_pixel_sizes=(0.29, 0.29, 0.29),
    ) -> Union[CellImagesResult, CellImagesError]:

        # Get the ultimate end save paths for this cell

        if original_structure:
            cell_image_3d_save_path = (
                cell_images_3d_dir
                / f"{row.CellId}_Original_{row.OriginalTaggedStructure}.tiff"
            )
            cell_image_2d_all_proj_save_path = (
                cell_images_2d_all_proj_dir
                / f"{row.CellId}_Original_{row.OriginalTaggedStructure}.png"
            )
            cell_image_2d_yx_proj_save_path = (
                cell_images_2d_yx_proj_dir
                / f"{row.CellId}_Original_{row.OriginalTaggedStructure}.png"
            )
        else:
            cell_image_3d_save_path = (
                cell_images_3d_dir / f"{row.CellId}_{row.GeneratedStructureName_i}_"
                f"{row.GeneratedStructureInstance_i}.tiff"
            )
            cell_image_2d_all_proj_save_path = (
                cell_images_2d_all_proj_dir
                / f"{row.CellId}_{row.GeneratedStructureName_i}_"
                f"{row.GeneratedStructureInstance_i}.png"
            )
            cell_image_2d_yx_proj_save_path = (
                cell_images_2d_yx_proj_dir
                / f"{row.CellId}_{row.GeneratedStructureName_i}_"
                f"{row.GeneratedStructureInstance_i}.png"
            )

        log.info(f"{cell_image_3d_save_path}")

        channels = [
            "nucleus_segmentation",
            "membrane_segmentation",
            "dna",
            "membrane",
            "structure",
            "brightfield",
        ]

        # Check skip
        if (
            not overwrite
            # Only skip if all images exist for this cell
            and all(
                p.is_file()
                for p in [
                    cell_image_3d_save_path,
                    cell_image_2d_all_proj_save_path,
                    cell_image_2d_yx_proj_save_path,
                ]
            )
        ):
            log.info(f"Skipping single cell image generation for CellId: {row.CellId}")
            return CellImagesResult(
                row.GeneratedStructureName_i,
                row.GeneratedStructureInstance_i,
                cell_image_3d_save_path,
                cell_image_2d_all_proj_save_path,
                cell_image_2d_yx_proj_save_path,
            )

        # Overwrite or didn't exist
        log.info(
            f"Beginning single cell image generation for : {cell_image_3d_save_path}"
        )

        # Choose either the segmentation channels or gfp
        if projection_channels == "seg":
            channel_subset = [
                "nucleus_segmentation",
                "membrane_segmentation",
            ]
        else:
            channel_subset = [
                "dna",
                "membrane",
            ]

        # Choose original gfp structure or generated gfp structure
        if original_structure:
            channel_subset.append("structure")

            image = image[[channels.index(target) for target in channel_subset]]
            combined_image = image
        else:
            channel_subset.append("generated_structure")

            image = image[[channels.index(target) for target in channel_subset[0:2]]]

            # Get generated image, it is already CZYX
            gen_image = AICSImage(row.GeneratedStructuePath_i)
            gen_image = gen_image.data

            # Remove scene and time information
            gen_image = gen_image[0, 0, :, :, :, :]

            # Append dna and membrane source to generate structure
            combined_image = np.append(image, gen_image, axis=0)

        # Store 3D image
        # Reduce size
        crop_3d = combined_image * 255
        crop_3d = crop_3d.astype(np.uint8)

        # Save to OME-TIFF
        with OmeTiffWriter(cell_image_3d_save_path, overwrite_file=False) as writer:
            writer.save(
                crop_3d,
                dimension_order="CZYX",
                channel_names=channel_subset,
                pixels_physical_size=desired_pixel_sizes,
            )

        # If segmentations were chosen
        # Do max projecting the way Matheus does it
        # (use appropriate z slice range for each structure)
        if projection_channels == "seg":

            log.info(f"Generating thumbnail contours for: {cell_image_3d_save_path}")

            zslice_range = getattr(ZSliceRanges, row.Gene)
            thumbnail = []
            for proj in range(3):
                thumbnail.append(
                    get_thumbnail(
                        input_raw=combined_image[2],
                        segs=combined_image[0:2],
                        normalize="pct",
                        proj=proj,
                        # This is a range of z slices
                        mode_raw=zslice_range[0],
                        # This is either "nuc" or "memb" to
                        # help determine the correct zslice
                        reference=zslice_range[1],
                        radii=(50, 50),
                    )
                )
            fig, axs = plt.subplots(
                2, 2, figsize=(6, 6), gridspec_kw={"hspace": 0.0, "wspace": 0.0}
            )
            axs[0, 1].imshow(thumbnail[2])
            axs[0, 0].imshow(thumbnail[0])
            axs[1, 0].imshow(thumbnail[1])
            for ax in axs.flatten():
                ax.axis("off")
            fig.savefig(cell_image_2d_all_proj_save_path)

        # If gfp is chosen, just max project each gfp channel
        else:
            colors = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
            # Get all axes projection image
            all_proj = proc.imgtoprojection(
                combined_image,
                proj_all=True,
                proj_method=projection_method,
                local_adjust=False,
                global_adjust=True,
                colors=colors,
            )

            # Convert to YXC for PNG writing
            all_proj = transforms.transpose_to_dims(all_proj, "CYX", "YXC")

            # Drop size to uint8
            all_proj = all_proj.astype(np.uint8)

            # Save to PNG
            imwrite(cell_image_2d_all_proj_save_path, all_proj)

            # Get YX axes projection image
            yx_proj = proc.imgtoprojection(
                combined_image,
                proj_all=False,
                proj_method=projection_method,
                local_adjust=False,
                global_adjust=True,
                colors=colors,
            )

            # Convert to YXC for PNG writing
            yx_proj = transforms.transpose_to_dims(yx_proj, "CYX", "YXC")

            # Drop size to uint8
            yx_proj = yx_proj.astype(np.uint8)

            # Save to PNG
            imwrite(cell_image_2d_yx_proj_save_path, yx_proj)

        log.info(
            f"Completed single cell image generation for: {cell_image_3d_save_path}"
        )

        # Return ready to save image
        return CellImagesResult(
            row.GeneratedStructureName_i,
            row.GeneratedStructureInstance_i,
            cell_image_3d_save_path,
            cell_image_2d_all_proj_save_path,
            cell_image_2d_yx_proj_save_path,
        )

    @staticmethod
    def _make_html_report(
        dataset: pd.DataFrame,
        cell_images_2d_all_proj_dir: Union[str, Path],
        cell_images_2d_yx_proj_dir: Union[str, Path],
        ncells: int,
    ):

        log.info("Beginning html report generation")

        html_dataframe = {"HTMLSavePath": [], "StructureName": []}

        for structure_name in dataset["GeneratedStructureName_i"].unique():
            this_html_save_path = (
                f"./local_staging/makediagnosticsheet/main_{structure_name}.html"
            )
            htmlwriter = HTMLWriter(this_html_save_path)

            htmlwriter.heading(
                f"These are {ncells - 1} IC generated {structure_name}: "
                "Original structure is "
                f"{dataset.head(1)['OriginalTaggedStructure'].item()}"
            )

            sub_dataset = dataset.loc[
                dataset["GeneratedStructureName_i"] == structure_name
            ]

            # htmlwriter.set_png_folder(cell_images_2d_all_proj_dir)
            htmlwriter.set_png_folder(Path("./cell_images_2d_all_proj/"))

            count = 0

            for index, row in sub_dataset.iterrows():
                if row.GeneratedStructureInstance_i == 0:
                    this_label = (
                        f"CellID_{row.CellId}_OriginalStructure_"
                        + f"{row.OriginalTaggedStructure}"
                    )
                else:
                    this_label = (
                        f"CellID_{row.CellId}_GeneratedImage_"
                        + f"{row.GeneratedStructureName_i}_"
                        + f"{row.GeneratedStructureInstance_i}"
                    )

                this_filename = row.CellImage2DAllProjectionsPath
                this_filename = str(this_filename).split("/", 4)[-1]

                htmlwriter.imagelink(
                    filename=f"{this_filename}",
                    link=".",
                    width=128,
                    label=f"{this_label}",
                )

                count += 1

            htmlwriter.close()

            html_dataframe["HTMLSavePath"].append(this_html_save_path)
            html_dataframe["StructureName"].append(structure_name)

        log.info("Completed html report generation")

        html_dataframe = pd.DataFrame(html_dataframe)

        return html_dataframe

    @log_run_params
    def run(
        self,
        dataset: Union[str, Path, pd.DataFrame, dd.DataFrame],
        cell_ceiling_adjustment: int = 7,
        bounding_box_percentile: float = 95.0,
        projection_method: str = "max",
        projection_channels: str = "seg",
        distributed_executor_address: Optional[str] = None,
        batch_size: Optional[int] = None,
        structs: Optional[List] = None,
        ncells: Optional[int] = None,
        overwrite: bool = False,
        current_pixel_sizes=(0.10833333333333332, 0.10833333333333332, 0.29,),
        desired_pixel_sizes=(0.29, 0.29, 0.29),
        **kwargs,
    ):
        """
        Provided a dataset of cell features and standardized FOV images, generate 3D
        single cell crops and 2D projections.

        Parameters
        ----------
        dataset: Union[str, Path, pd.DataFrame, dd.DataFrame]
            The primary cell dataset to generate 3D single cell images for.

            Example dataset: /allen/aics/modeling/ritvik/projects/
            cvapipe/local_staging/generategfpinstantiations_tmp/images_CellID_109017/manifest.csv

        cell_ceiling_adjustment: int
            The adjust to use for raising the cell shape ceiling. If <= 0, this will be
            ignored and cell data will be selected but not adjusted.
            Default: 7

        bounding_box_percentile: float
            A float used to generate the actual bounding box for all cells by finding
            provided percentile of all cell image sizes.
            Default: 95.0

        projection_method: str
            The method to use for generating the flat projection.
            Default: max

            More details:
            https://allencellmodeling.github.io/aicsimageprocessing/aicsimageprocessing.html#aicsimageprocessing.imgToProjection.imgtoprojection

        projection_channels: str
            The channels to max project. If "seg", then will make contours 
            the dna and memb seg channels
            along with the structure gfp. Else will max project gfp 
            of dna, memb and struct
        
        distributed_executor_address: Optional[str]
            An optional executor address to pass to some computation engine.
            Default: None

        batch_size: Optional[int]
            An optional batch size to process n features at a time.
            Default: None (Process all at once)

        overwrite: bool
            If this step has already partially or completely run, should it overwrite
            the previous files or not.
            Default: False (Do not overwrite or regenerate files)

        Returns
        -------
        manifest_save_path: Path
            Path to the produced manifest with the various cell image path fields added.
        """
        # Handle dataset provided as string or path
        if isinstance(dataset, (str, Path)):
            dataset = Path(dataset).expanduser().resolve(strict=True)

            # Read dataset
            dataset = pd.read_csv(dataset)

        # Select ncells for each group
        dataset = (
            dataset.groupby(["GeneratedStructureName_i"])
            .apply(
                lambda x: x.sort_values(
                    ["GeneratedStructureInstance_i"], ascending=True
                )
            )
            .reset_index(drop=True)
        )

        if ncells:
            # select top N rows within each continent
            dataset = dataset.groupby("GeneratedStructureName_i").head(ncells)
        if structs:
            # Select structures we care about
            dataset = dataset.loc[dataset["GeneratedStructureName_i"].isin(structs)]

        # Create save directories'
        cell_images_3d_dir = self.step_local_staging_dir / "cell_images_3d"
        cell_images_2d_all_proj_dir = (
            self.step_local_staging_dir / "cell_images_2d_all_proj"
        )
        cell_images_2d_yx_proj_dir = (
            self.step_local_staging_dir / "cell_images_2d_yx_proj"
        )

        cell_images_3d_dir.mkdir(exist_ok=True)
        cell_images_2d_all_proj_dir.mkdir(exist_ok=True)
        cell_images_2d_yx_proj_dir.mkdir(exist_ok=True)

        # Process each row
        with DistributedHandler(distributed_executor_address) as handler:

            # Reset index because we want to do image processing
            # only on the firt row
            dataset.reset_index(inplace=True)

            # Start processing
            # Bounding box is for the original source images for Greg's 3D trained model
            # Num of channels is 7 for all
            num_of_channels = 7
            bbox = [64, 160, 96]
            log.info(f"Using hard coded bounding box with ZYX dimensions: {bbox}.")

            bbox = [num_of_channels] + list(bbox)
            bbox_results = [bbox]

            # Compute bounding box with percentile
            bbox_results = np.array(bbox_results)
            bounding_box = np.percentile(bbox_results, bounding_box_percentile, axis=0)
            bounding_box = np.ceil(bounding_box)

            # First normalize images
            normalized_img = handler.batched_map(
                self._normalize_images,
                *zip(*list(dataset.iterrows())),
                [cell_ceiling_adjustment for i in range(len(dataset))],
                [bounding_box for i in range(len(dataset))],
                [current_pixel_sizes for i in range(len(dataset))],
                [desired_pixel_sizes for i in range(len(dataset))],
                batch_size=batch_size,
            )

            for j, i in enumerate(normalized_img):
                if i is not None:
                    img = i
                else:
                    normalized_img[j] = img

            plot_original_structure = dataset["GeneratedStructureInstance_i"] == 0
            plot_original_structure = plot_original_structure.values

            # Generate bounded arrays
            results = handler.batched_map(
                self._generate_single_cell_images,
                # Convert dataframe iterrows into two lists of items to iterate over
                # One list will be row index
                # One list will be the pandas series of every row
                *zip(*list(dataset.iterrows())),
                [i for i in normalized_img],
                # Pass the other parameters as list of the same thing for each
                # mapped function call
                [projection_method for i in range(len(dataset))],
                [projection_channels for i in range(len(dataset))],
                [cell_images_3d_dir for i in range(len(dataset))],
                [cell_images_2d_all_proj_dir for i in range(len(dataset))],
                [cell_images_2d_yx_proj_dir for i in range(len(dataset))],
                [overwrite for i in range(len(dataset))],
                [i for i in plot_original_structure],
                [current_pixel_sizes for i in range(len(dataset))],
                [desired_pixel_sizes for i in range(len(dataset))],
                batch_size=batch_size,
            )

        # Generate single cell images dataset rows
        single_cell_images_dataset = []
        errors = []
        for r in results:
            if isinstance(r, CellImagesResult):
                single_cell_images_dataset.append(
                    {
                        DatasetFields.GeneratedStructureName: r.generated_struct_name,
                        DatasetFields.GeneratedStructureInstance: r.gen_struct_inst,
                        DatasetFields.CellImage3DPath: r.path_3d,
                        DatasetFields.CellImage2DAllProjectionsPath: r.path_2d_all_proj,
                        DatasetFields.CellImage2DYXProjectionPath: r.path_2d_yx_proj,
                    }
                )
            else:
                errors.append({DatasetFields.CellId: r.cell_id, "Error": r.error})

        # Convert features paths rows to dataframe
        single_cell_images_dataset = pd.DataFrame(single_cell_images_dataset)

        self.manifest = dataset.merge(
            single_cell_images_dataset,
            on=[
                DatasetFields.GeneratedStructureName,
                DatasetFields.GeneratedStructureInstance,
            ],
        )

        # Now we have single cell image dataset, make diagnostic sheet
        # Process each row

        html_dataframe = self._make_html_report(
            self.manifest,
            cell_images_2d_all_proj_dir,
            cell_images_2d_yx_proj_dir,
            ncells,
        )

        # Save manifest to CSV
        html_report_save_path = self.step_local_staging_dir / "manifest.csv"
        html_dataframe.to_csv(html_report_save_path, index=False)

        # Save errored cells to JSON
        with open(self.step_local_staging_dir / "errors.json", "w") as write_out:
            json.dump(errors, write_out)

        return html_report_save_path
