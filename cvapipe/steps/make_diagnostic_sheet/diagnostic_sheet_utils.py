import logging
import datetime

from typing import List, Optional, Tuple

import aicsimageprocessing as proc
import dask.array as da
import numpy as np
from scipy.signal import fftconvolve as convolve
from aicsimageio import AICSImage, types
from aicsimageio.writers import OmeTiffWriter

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Channels:
    NucleusSegmentation = "nucleus_segmentation"
    MembraneSegmentation = "membrane_segmentation"
    DNA = "dna"
    Membrane = "membrane"
    Structure = "structure"
    Brightfield = "brightfield"
    DefaultOrderList = [
        NucleusSegmentation,
        MembraneSegmentation,
        DNA,
        Membrane,
        Structure,
        Brightfield,
    ]


def get_normed_image_array(
    raw_image: types.ImageLike,
    nucleus_seg_image: types.ImageLike,
    membrane_seg_image: types.ImageLike,
    dna_channel_index: int,
    membrane_channel_index: int,
    structure_channel_index: int,
    brightfield_channel_index: int,
    nucleus_seg_channel_index: int,
    membrane_seg_channel_index: int,
    current_pixel_sizes: Optional[Tuple[float]] = None,
    desired_pixel_sizes: Optional[Tuple[float]] = None,
) -> Tuple[np.ndarray, List[str], Tuple[float]]:
    """
    Provided the original raw image, and a nucleus and membrane segmentation, construct
    a standardized, ordered, and normalized array of the images.

    Parameters
    ----------
    raw_image: types.ImageLike
        A filepath to the raw imaging data. The image should be 4D and include
        channels for DNA, Membrane, Structure, and Transmitted Light.

    nucleus_seg_image: types.ImageLike
        A filepath to the nucleus segmentation for the provided raw image.

    membrane_seg_image: types.ImageLike
        A filepath to the membrane segmentation for the provided raw image.

    dna_channel_index: int
        The index in channel dimension in the raw image that stores DNA data.

    membrane_channel_index: int
        The index in the channel dimension in the raw image that stores membrane data.

    structure_channel_index: int
        The index in the channel dimension in the raw image that stores structure data.

    brightfield_channel_index: int
        The index in the channel dimension in the raw image that stores the brightfield
        data.

    nucleus_seg_channel_index: int
        The index in the channel dimension in the nucleus segmentation image that stores
        the segmentation.

    membrane_seg_channel_index: int
        The index in the channel dimension in the membrane segmentation image that
        stores the segmentation.

    current_pixel_sizes: Optioal[Tuple[float]]
        The current physical pixel sizes as a tuple of the raw image.
        Default: None (`aicsimageio.AICSImage.get_physical_pixel_size` on the raw image)

    desired_pixel_sizes: Optional[Tuple[float]]
        The desired physical pixel sizes as a tuple to scale all images to.
        Default: None (scale all images to current_pixel_sizes if different)

    Returns
    -------
    normed: np.ndarray
        The normalized images stacked into a single CYXZ numpy ndarray.

    channels: List[str]
        The standardized channel names for the returned array.

    pixel_sizes: Tuple[float]
        The physical pixel sizes of the returned image in XYZ order.

    Notes
    -----
    The original version of this function can be found at:
    https://aicsbitbucket.corp.alleninstitute.org/projects/MODEL/repos/image_processing_pipeline/browse/aics_single_cell_pipeline/utils.py#9
    """
    # Construct image objects
    raw = AICSImage(raw_image)
    nuc_seg = AICSImage(nucleus_seg_image)
    memb_seg = AICSImage(membrane_seg_image)

    # Preload image data
    raw.data
    nuc_seg.data
    memb_seg.data

    # Get default current and desired pixel sizes
    if current_pixel_sizes is None:
        current_pixel_sizes = raw.get_physical_pixel_size()

    # Default desired to be the same pixel size
    if desired_pixel_sizes is None:
        desired_pixel_sizes = current_pixel_sizes

    # Select the channels
    channel_indices = [
        dna_channel_index,
        membrane_channel_index,
        structure_channel_index,
        brightfield_channel_index,
    ]
    selected_channels = [
        raw.get_image_dask_data("YXZ", S=0, T=0, C=index) for index in channel_indices
    ]

    # Combine selections and get numpy array
    raw = da.stack(selected_channels).compute()

    # Convert pixel sizes to numpy arrays
    current_pixel_sizes = np.array(current_pixel_sizes)
    desired_pixel_sizes = np.array(desired_pixel_sizes)

    # Only resize raw image if desired pixel sizes is different from current
    if not np.array_equal(current_pixel_sizes, desired_pixel_sizes):
        scale_raw = current_pixel_sizes / desired_pixel_sizes
        raw = np.stack([proc.resize(channel, scale_raw, "bilinear") for channel in raw])

    # Prep segmentations
    nuc_seg = nuc_seg.get_image_data("YXZ", S=0, T=0, C=nucleus_seg_channel_index)
    memb_seg = memb_seg.get_image_data("YXZ", S=0, T=0, C=membrane_seg_channel_index)

    # We do not assume that the segmentations are the same size as the raw
    # Resize the segmentations to match the raw
    # We drop the channel dimension from the raw size retrieval
    raw_size = np.array(raw.shape[1:]).astype(float)
    nuc_size = np.array(nuc_seg.shape).astype(float)
    memb_size = np.array(memb_seg.shape).astype(float)
    scale_nuc = raw_size / nuc_size
    scale_memb = raw_size / memb_size

    # Actual resize
    nuc_seg = proc.resize(nuc_seg, scale_nuc, method="nearest")
    memb_seg = proc.resize(memb_seg, scale_memb, method="nearest")

    # Normalize images
    normalized_images = []
    for i, index in enumerate(channel_indices):
        if index == brightfield_channel_index:
            norm_method = "trans"
        else:
            norm_method = "img_bg_sub"

        # Normalize and append
        normalized_images.append(proc.normalize_img(raw[i], method=norm_method))

    # Stack all together
    img = np.stack([nuc_seg, memb_seg, *normalized_images])
    channel_names = Channels.DefaultOrderList

    return img, channel_names, tuple(desired_pixel_sizes)


def im_write(im, path):

    im = im.cpu().detach().numpy().transpose(3, 0, 1, 2)

    im = im.copy(order="C")

    with OmeTiffWriter(path, overwrite_file=True) as writer:
        writer.save(im)


def select_and_adjust_segmentation_ceiling(
    image: np.ndarray, cell_index: int, cell_ceiling_adjustment: int = 0
) -> np.ndarray:
    """
    Select and adjust the cell shape "ceiling" for a specific cell in the provided
    image.

    Parameters
    ----------
    image: np.ndarray
        The 4D, CYXZ, image numpy ndarray output from `get_normed_image_array`.

    cell_index: int
        The integer index for the target cell.

    cell_ceiling_adjustment: int
        The adjust to use for raising the cell shape ceiling. If <= 0, this will be
        ignored and cell data will be selected but not adjusted.
        Default: 0

    Returns
    -------
    adjusted: np.ndarray
        The image with the membrane segmentation adjusted for ceiling shape correction.

    Notes
    -----
    The original version of this function can be found at:
    https://aicsbitbucket.corp.alleninstitute.org/projects/MODEL/repos/image_processing_pipeline/browse/aics_single_cell_pipeline/utils.py#83
    """
    # Select only the data in the first two channels (the segmentation channels)
    # where the data matches the provided cell index
    image[0:2] = image[0:2] == cell_index

    # Because they are conservatively segmented,
    # we raise the "ceiling" of the cell shape

    # This is the so-called "roof-augmentation" that Greg (@gregjohnso) invented to
    # handle the bad "roof" in old membrane segmentations.
    #
    # Specially, because the photobleaching, the signal near the top is very weak.
    # Then, the membrane segmentation stops earlier (in terms of Z position) than the
    # truth. For some structures living near the top of the cell, like mitochodira, the
    # structure segmentation may be out of the membrane segmentation, as the membrane
    # segmentation is "shorter" than it should be, then the structure segmentation
    # will be mostly choped off and make the integrated cell model learn nothing.
    #
    # So, "roof-augmentation" is the method to fix the "shorter" membrane segmentation
    # issues.

    # Adjust image ceiling if adjustment is greater than zero
    if cell_ceiling_adjustment > 0:
        # Get the center of mass of the nucleus
        nuc_com = proc.get_center_of_mass(image[0])[-1]

        # Get the top of the membrane
        memb_top = np.where(np.sum(np.sum(image[1], axis=0), axis=0))[0][-1]

        # Get the halfway point between the two
        start = int(np.floor((nuc_com + memb_top) / 2))

        # Get the shape of the cell from the membrane segmentation
        cell_shape = image[1, :, :, start:]

        # Adjust cell shape "ceiling" using the adjustment integer provided
        start_ind = int(np.floor(cell_ceiling_adjustment)) - 1
        imf = np.zeros([1, 1, cell_ceiling_adjustment * 2 - 1])
        imf[:, :, start_ind:] = 1
        cell_shape = convolve(cell_shape, imf, mode="same") > 1e-8

        # Set the image data with the new cell shape data
        image[1, :, :, start:] = cell_shape

    return image


def crop_raw_channels_with_segmentation(
    image: np.ndarray, channels: List[str]
) -> np.ndarray:
    """
    Crop imaging data in raw channels using a provided selected full field of with a
    target cell in the segmentation channels.

    Parameters
    ----------
    image: np.ndarray
        The 4D, CYXZ, image numpy ndarray output from
        `select_and_adjust_segmentation_ceiling`.

    channels: List[str]
        The channel names for the provided image.
        The channels output from `get_normed_image_array`.

    Returns
    -------
    cropped: np.ndarray
        A 4D numpy ndarray with CYXZ dimensions in the same order as provided.
        The raw DNA channel has been cropped using the nucleus segmentation.
        All other raw channels have been cropped using the membrane segmentation.

    Notes
    -----
    The original version of this function can be found at:
    https://aicsbitbucket.corp.alleninstitute.org/projects/MODEL/repos/image_processing_pipeline/browse/aics_single_cell_pipeline/utils.py#114
    """
    # Select segmentation indicies
    nuc_ind = np.array(channels) == Channels.NucleusSegmentation
    memb_ind = np.array(channels) == Channels.MembraneSegmentation

    # Select DNA and all other indicies
    dna_ind = np.array(channels) == Channels.DNA
    other_channel_inds = np.ones(len(channels))
    other_channel_inds[nuc_ind | memb_ind | dna_ind] = 0

    # Crop DNA channel with the nucleus segmentation
    image[dna_ind] = image[dna_ind] * image[nuc_ind]

    # All other channels are cropped using membrane segmentation
    for i in np.where(other_channel_inds)[0]:
        image[i] = image[i] * image[memb_ind]

    return image


class HTMLWriter:

    """
        Class to write information in HTML format
    """

    def __init__(self, path):

        self.out = open(path, "w")
        self.html_init()
        self.folder_tif = ""
        self.folder_png = ""

    def set_tif_folder(self, path):
        self.folder_tif = path

    def set_png_folder(self, path):
        self.folder_png = path

    def print(self, txt):
        print(txt, file=self.out)

    def html_init(self):
        self.print("<!DOCTYPE html>")
        self.print("<html>")
        self.print("<head>")
        self.print("<style>")
        self.print("img {width: 100%;}")
        self.print(
            "h1 { color: #111; font-family: 'Helvetica Neue', sans-serif; font-size: 32px; font-weight: bold; letter-spacing: -1px; line-height: 1; text-align: left; }"
        )
        self.print(
            "h2 { color: #111; font-family: 'Open Sans', sans-serif; font-size: 24px; font-weight: 300; line-height: 32px; margin: 0 0 16px; text-align: left; }"
        )
        self.print(
            "h3 { color: #685206; font-family: 'Helvetica Neue', sans-serif; font-size: 14px; line-height: 24px; margin: 0 0 24px; text-align: justify; text-justify: inter-word; }"
        )
        self.print("</style>")
        self.print("</head>")
        self.print("<body>")
        self.print(
            f"Current date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def heading(self, txt, heading=1):
        self.print(f"<h{heading}>{txt}</h{heading}>")

    def imagelink(self, filename, link, width, label=None):
        self.print(
            f'<div style="width: {width}px; float: left; margin: 0px; padding: 0px">'
        )
        self.print(f'<a href="{self.folder_tif}/{link}" target="_blank">')
        label = label if label is not None else ""
        self.print(f'<img src="{self.folder_png}/{filename}" title="{label}">')
        self.print("</a>")
        self.print("</div>")

    def br(self):
        self.print('<div style="clear:both"></div>')

    def close(self):
        self.br()
        self.print(
            f"Current date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.print("</body>")
        self.print("</html>")
        self.out.close()
