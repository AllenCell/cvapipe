import numpy as np
from scipy.ndimage import zoom
from skimage.morphology import dilation, ball
from skimage.measure import label


def getLargestCC(labels):
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def find_true_edge_cells(mem_seg_whole_valid):

    mem_coverage = np.amax(mem_seg_whole_valid, axis=0)
    FOV_size = mem_coverage.shape[0] * mem_coverage.shape[1]
    mem_coverage_ratio = np.count_nonzero(mem_coverage > 0) / FOV_size
    if mem_coverage_ratio > 0.95:
        # not enough plate area
        return []

    # extract body chunk
    colony_top = mem_seg_whole_valid.shape[0] - 1
    for zz in np.arange(mem_seg_whole_valid.shape[0] - 1,
                        mem_seg_whole_valid.shape[0] // 2 + 1, -1):
        if np.count_nonzero(mem_seg_whole_valid[zz, :, :] > 0) / FOV_size \
           > 0.5 * mem_coverage_ratio:
            colony_top = zz 
            break

    colony_bottom = 0
    for zz in np.arange(3, mem_seg_whole_valid.shape[0] // 2):
        if np.count_nonzero(mem_seg_whole_valid[zz, :, :] > 0) / FOV_size \
           > 0.5 * mem_coverage_ratio:
            colony_bottom = zz
            break

    # make sure not super flat 
    if colony_top - colony_bottom < 5:
        return []

    # find the list of true edge cells
    colony_middle = (colony_top - colony_bottom) // 2 + colony_bottom
    body_chunk = mem_seg_whole_valid[colony_middle - 2:colony_middle + 3, :, :]

    # extract cover slip object
    bg_in_body_chunk = body_chunk == 0
    bg_label_in_body_chunk = label(bg_in_body_chunk)
    outer_bg = getLargestCC(bg_label_in_body_chunk)

    # find cells touching cover slip
    outer_bg_cover = dilation(outer_bg, ball(5))
    edge_idx = list(np.unique(body_chunk[outer_bg_cover > 0]))

    return edge_idx


def build_one_cell_for_classification(crop_raw, mem_seg, down_ratio=0.5):

    # raw image normalization
    img_raw = crop_raw[0, :, 0:2, :, :].astype(np.float32)
    img_raw = np.transpose(img_raw, (1, 0, 2, 3))
    for dim in range(0, 2):
        img_ch = img_raw[dim, :, :, :].copy()
        low = np.percentile(img_ch, 0.05)
        upper = np.percentile(img_ch, 99.5)
        img_ch[img_ch > upper] = upper
        img_ch[img_ch < low] = low
        img_ch = (img_ch - low) / (upper - low)
        img_raw[dim, :, :, :] = img_ch

    # load seg
    mem_seg_tight = dilation(mem_seg > 0, ball(3))

    z_range = np.where(np.any(mem_seg_tight, axis=(1, 2)))
    y_range = np.where(np.any(mem_seg_tight, axis=(0, 2)))
    x_range = np.where(np.any(mem_seg_tight, axis=(0, 1)))
    z_range = z_range[0]
    y_range = y_range[0]
    x_range = x_range[0]

    roi = [max(z_range[0] - 2, 0), min(z_range[-1] + 4, mem_seg_tight.shape[0]),
           max(y_range[0] - 5, 0), min(y_range[-1] + 5, mem_seg_tight.shape[1]),
           max(x_range[0] - 5, 0), min(x_range[-1] + 5, mem_seg_tight.shape[2])]

    mem_seg_tight = mem_seg_tight[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]

    mem_img = img_raw[1, roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
    dna_img = img_raw[0, roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]

    mem_seg_tight = zoom(mem_seg_tight, down_ratio, order=0)
    mem_img = zoom(mem_img, down_ratio, order=2)
    dna_img = zoom(dna_img, down_ratio, order=2)

    mem_img[mem_seg_tight == 0] = 0
    dna_img[mem_seg_tight == 0] = 0

    # merge seg and raw
    img_out = np.stack((dna_img, mem_img), axis=0)

    return img_out


def euc_dist_3d(p1, p2):
    dist = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**.5
    return dist


def euc_dist_2d(p1, p2):
    dist = ((p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**.5
    return dist


def overlap_area(p1, p2):
    matrix = np.zeros((3, 7, 7))
    matrix[1, :, :] = 1
    d1 = dilation(p1, selem=matrix)
    d2 = dilation(p2, selem=matrix)
    area = np.count_nonzero(np.logical_and(d1, d2))
    return area
