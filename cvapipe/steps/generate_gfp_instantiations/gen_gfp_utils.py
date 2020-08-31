import os

import numpy as np
import logging
import torch

# import sys

# # insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, "/allen/aics/modeling/ritvik/projects/pytorch_integrated_cell/")

from integrated_cell.networks.ref_target_autoencoder import Autoencoder
from integrated_cell import utils

from aicsimageio.writers import OmeTiffWriter

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def im_write(im, path):

    im = im.cpu().detach().numpy().transpose(3, 0, 1, 2)

    im = im.copy(order="C")

    with OmeTiffWriter(path, overwrite_file=True) as writer:
        writer.save(im)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def SetupTheAutoencoder(GPUID, REF_MODEL_KWARGS=None, TARG_MODEL_KWARGS=None):

    # Prep GPUs
    gpu_ids = [4]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(ID) for ID in gpu_ids])
    if len(gpu_ids) == 1:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    torch.cuda.empty_cache()

    # Load models
    log.info("Beginning model load")

    if not REF_MODEL_KWARGS:
        REF_MODEL_KWARGS = dict(
            model_dir="/allen/aics/modeling/gregj/results/integrated_cell"
            + "/test_cbvae_3D_avg_inten/2019-11-27-22:27:04",
            parent_dir="/allen/aics/modeling/gregj/results/integrated_cell/",
            suffix="_94544",
        )

    if not TARG_MODEL_KWARGS:
        TARG_MODEL_KWARGS = dict(
            parent_dir="/allen/aics/modeling/gregj/results/integrated_cell/",
            model_dir="/allen/aics/modeling/gregj/results/integrated_cell"
            + "/test_cbvae_3D_avg_inten/2019-10-22-15:24:09/",
            suffix="_93300",
        )

    networks_ref, dp_ref, args_ref = utils.load_network_from_dir(
        REF_MODEL_KWARGS["model_dir"],
        REF_MODEL_KWARGS["parent_dir"],
        suffix=REF_MODEL_KWARGS["suffix"],
    )

    ref_enc = networks_ref["enc"]
    ref_dec = networks_ref["dec"]

    networks_targ, dp_target, args_target = utils.load_network_from_dir(
        TARG_MODEL_KWARGS["model_dir"],
        TARG_MODEL_KWARGS["parent_dir"],
        suffix=TARG_MODEL_KWARGS["suffix"],
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
