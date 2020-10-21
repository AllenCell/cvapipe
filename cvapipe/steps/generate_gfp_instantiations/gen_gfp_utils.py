import os

import numpy as np
import logging
import torch

from integrated_cell.networks.ref_target_autoencoder import Autoencoder
from integrated_cell import utils

from aicsimageio.writers import OmeTiffWriter

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def im_write(im, path):
    """
    Takes CZYX torch tensor and writes ome.tiff to path
    """

    im = im.cpu().detach().numpy().transpose(3, 0, 1, 2)

    im = im.copy(order="C")

    with OmeTiffWriter(path, overwrite_file=True) as writer:
        writer.save(im)


def setup_the_autoencoder(gpu_id, ref_model_kwargs=None, targ_model_kwargs=None):
    """
    Load Greg's 3D trained model
    and return the autoencoder (train set to False),
    the dataprovider, and the classes and class names
    """

    # Prep GPUs
    gpu_ids = [gpu_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(ID) for ID in gpu_ids])
    if len(gpu_ids) == 1:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    torch.cuda.empty_cache()

    # Load models
    log.info("Beginning model load")

    if not ref_model_kwargs:
        ref_model_kwargs = dict(
            model_dir="/allen/aics/modeling/gregj/results/integrated_cell"
            + "/test_cbvae_3D_avg_inten/2019-11-27-22:27:04",
            parent_dir="/allen/aics/modeling/gregj/results/integrated_cell/",
            suffix="_94544",
        )

    if not targ_model_kwargs:
        targ_model_kwargs = dict(
            parent_dir="/allen/aics/modeling/gregj/results/integrated_cell/",
            model_dir="/allen/aics/modeling/gregj/results/integrated_cell"
            + "/test_cbvae_3D_avg_inten/2019-10-22-15:24:09/",
            suffix="_93300",
        )

    networks_ref, dp_ref, args_ref = utils.load_network_from_dir(
        ref_model_kwargs["model_dir"],
        ref_model_kwargs["parent_dir"],
        suffix=ref_model_kwargs["suffix"],
    )

    ref_enc = networks_ref["enc"]
    ref_dec = networks_ref["dec"]

    networks_targ, dp_target, args_target = utils.load_network_from_dir(
        targ_model_kwargs["model_dir"],
        targ_model_kwargs["parent_dir"],
        suffix=targ_model_kwargs["suffix"],
    )

    target_enc = networks_targ["enc"]
    target_dec = networks_targ["dec"]

    mode = "test"
    dp = dp_target
    u_classes, class_inds = np.unique(
        dp.get_classes(np.arange(0, dp.get_n_dat(mode)), mode), return_inverse=True
    )
    u_class_names = dp.label_names[u_classes]

    ae = Autoencoder(ref_enc, ref_dec, target_enc, target_dec)
    ae.train(False)
    ae = ae.cuda()

    log.info("End model load")

    return u_class_names, u_classes, dp, ae
