# --------------------------------------------------------
# Who_where
# Copyright (c) 2016 University of Virginia
# Licensed under The MIT License [see LICENSE for details]
# Written by Fuwen Tan @ U.Va (2016)
# --------------------------------------------------------

import os, cv2
import os.path as osp
import numpy as np
from config import cfg


################################################################################
# Path
################################################################################
def maybe_create(dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

################################################################################
# BBox representations
################################################################################
def clip_boxes(boxes, width, height):
    """clip the boxes to make them valid."""
    boxes[:, 0] = np.maximum(boxes[:, 0], 0)
    boxes[:, 1] = np.maximum(boxes[:, 1], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], width - 1)
    boxes[:, 3] = np.minimum(boxes[:, 3], height - 1)
    return boxes.astype(np.int32)


def xywh_to_xyxy(boxes, width, height):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    w = boxes[:, 2]
    h = boxes[:, 3] #np.divide(w, boxes[:, 3])
    xmin = boxes[:, 0] - 0.5 * w + 1.0
    xmax = boxes[:, 0] + 0.5 * w
    ymin = boxes[:, 1] - h + 1.0 #boxes[:, 1] - 0.5 * h + 1.0
    ymax = boxes[:, 1] #boxes[:, 1] + 0.5 * h
    xyxy = np.vstack((xmin, ymin, xmax, ymax)).transpose()

    return clip_boxes(xyxy, width, height)


def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    y = boxes[:, 3] #0.5 * (boxes[:, 1] + boxes[:, 3])
    w = boxes[:, 2] - boxes[:, 0] + 1.0
    h = boxes[:, 3] - boxes[:, 1] + 1.0
    # r = np.divide(w, h)
    return np.vstack((x, y, w, h)).transpose()


def xyxy_areas(boxes):
    areas = np.multiply(boxes[:, 2] - boxes[:, 0] + 1, \
                        boxes[:, 3] - boxes[:, 1] + 1)
    return areas


def expand_xyxy(boxes, width, height, ratio=0.2):
    bw = boxes[:, 2] - boxes[:, 0] + 1.0
    bh = boxes[:, 3] - boxes[:, 1] + 1.0

    ox = (bw * ratio).astype(np.int)
    oy = (bh * ratio).astype(np.int)

    xyxy = boxes.copy()
    xyxy[:,0] -= ox; xyxy[:,1] -= oy
    xyxy[:,2] += ox; xyxy[:,3] += oy

    return clip_boxes(xyxy, width, height)


def normalize_xywh(boxes, width, height):
    # max_logr = np.log(cfg.ASPECT_RATIO_RANGE[1])
    # min_logr = np.log(cfg.ASPECT_RATIO_RANGE[0])

    x = np.divide(boxes[:, 0], float(width))
    y = np.divide(boxes[:, 1], float(height))
    w = np.divide(boxes[:, 2], float(width))
    h = np.divide(boxes[:, 3], float(height))
    # r = np.maximum(boxes[:, 3], cfg.ASPECT_RATIO_RANGE[0])
    # r = np.minimum(r,           cfg.ASPECT_RATIO_RANGE[1])
    # r = np.divide(np.log(r) - min_logr, max_logr - min_logr)

    return np.vstack((x, y, w, h)).transpose()


def denormalize_xywh(boxes, width, height):
    # max_logr = np.log(cfg.ASPECT_RATIO_RANGE[1])
    # min_logr = np.log(cfg.ASPECT_RATIO_RANGE[0])

    x = np.multiply(boxes[:, 0], float(width))
    y = np.multiply(boxes[:, 1], float(height))
    w = np.multiply(boxes[:, 2], float(width))
    h = np.multiply(boxes[:, 3], float(height))
    # r = np.multiply(boxes[:, 3], max_logr - min_logr) + min_logr
    # r = np.exp(r)

    return np.vstack((x, y, w, h)).transpose()


def ind2sub(indices, resolution):
    # indices: linear indices
    # resolution: [x_dim, y_dim]
    # output: normalized xy
    x = (indices %  resolution[0] + 0.5) / resolution[0]
    y = (indices // resolution[1] + 0.5) / resolution[1]
    return np.vstack((x, y)).transpose()


def sub2ind(subscripts, resolution):
    # subscripts: normalized subscript
    # resolution: [x_dim, y_dim]
    # output: linear indices
    scaled_xy = np.multiply(subscripts, np.array(resolution).reshape((1, 2)))
    scaled_xy = np.ceil(scaled_xy)
    scaled_xy = np.maximum(0, scaled_xy-1)
    indices = scaled_xy[:,0] + scaled_xy[:,1] * resolution[0]

    return indices.astype(np.int32)


def normH2ind(normH, bin_size):
    ind = np.floor(normH * bin_size + 0.5)
    return np.maximum(0, ind - 1)


def ind2normH(ind, bin_size):
    return ind/float(bin_size)


def indices_to_boxes(indices, resolution):
    # indices: [np_samples, 2], linear indices of boxes
    # resolution: resolution of the subscripts
    # output: normalized boxes

    sub_1 = ind2sub(indices[:, 0], resolution[:2])
    sub_2 = ind2sub(indices[:, 1], resolution[2:])

    return np.hstack((sub_1, sub_2)).astype(np.float)


def boxes_to_indices(boxes, resolution):
    # boxes: normalized boxes

    ind_1 = sub2ind(boxes[:,:2], resolution[:2])
    ind_2 = sub2ind(boxes[:,2:], resolution[2:])

    return np.vstack((ind_1, ind_2)).transpose()


def centers_to_rois(centers, encode_dims, decode_dims, radius=cfg.PEEPHOLE_RADIUS):
    nxy = ind2sub(centers, encode_dims)
    xy  = np.multiply(nxy, np.array(decode_dims).reshape((1,2))).astype(np.int)

    rois = np.zeros((centers.shape[0], decode_dims[0], decode_dims[1]), dtype=np.float)
    for i in range(centers.shape[0]):
        rois[i, (xy[i,1]-2*radius):(xy[i,1]+1), \
                (xy[i,0]-radius):(xy[i,0]+radius+1)] = 1.0

    return rois


def scene_volume_from_entry(entry, cur_box=None, n_cls=80):
    width  = entry['width']
    height = entry['height']

    vol = np.zeros((height, width, n_cls), dtype=np.float)
    boxes = np.array(entry['all_boxes']).reshape((-1,4))
    clses = np.array(entry['all_clses']).flatten()

    for i in range(boxes.shape[0]):
        bb = boxes[i].astype(np.int)
        if not (cur_box is None):
            diff = np.sum(np.absolute(bb - cur_box))
            if diff < 4:
                continue

        bb = boxes[i].astype(np.int)
        cls = clses[i] - 1
        vol[bb[1]:(bb[3]+1), bb[0]:(bb[2]+1), cls] = 255

    return vol


def scene_layout(width, height, boxes, clses, color_palette):

    colors = np.zeros((height, width, 3), dtype=np.float)
    counts = np.zeros((height, width),    dtype=np.float)


    for i in range(boxes.shape[0]):
        cur_color = np.zeros((height, width, 3))
        cur_count = np.zeros((height, width))

        bb = boxes[i].astype(np.int)
        cls = clses[i]
        rgb = np.array(color_palette[cls])
        cur_color[bb[1]:(bb[3]+1), bb[0]:(bb[2]+1), :] = rgb.reshape((1,1,3))
        cur_count[bb[1]:(bb[3]+1), bb[0]:(bb[2]+1)] = 1
        colors = colors + cur_color
        counts = counts + cur_count

    counts = counts + 1e-3
    average = np.divide(colors, np.expand_dims(counts, axis=-1))
    output = average.copy().astype(np.int)
    output[average > 255] = 255
    output[average < 0] = 0

    sum_output = np.sum(output, axis=-1)
    sum_output = sum_output.flatten()
    indices = np.where(sum_output < 0.5)[0]

    output = output.reshape((-1, 3))
    output[indices, :] = np.array(color_palette[0]).flatten()
    output = output.reshape((height, width, 3))

    return output


def scene_volume(width, height, boxes, clses, ex_box=None, n_cls=80):
    vol = np.zeros((height, width, n_cls), dtype=np.float)

    for i in range(boxes.shape[0]):
        bb = boxes[i].astype(np.int)
        if not (ex_box is None):
            diff = np.sum(np.absolute(bb - ex_box))
            if diff < 4:
                continue

        bb = boxes[i].astype(np.int)
        cls = clses[i] - 1
        vol[bb[1]:(bb[3]+1), bb[0]:(bb[2]+1), cls] = 255

    return vol


################################################################################
# Box transformation for offset regression
################################################################################
def bbox_transform(grid_xywh, gt_xywh):
    # Assume the input xywh have been normalized.
    dxdy = np.divide(gt_xywh[:,:2] - grid_xywh[:,:2], grid_xywh[:,2:])
    dxdy = dxdy.transpose()
    dw   = np.log(gt_xywh[:, 2] / grid_xywh[:, 2])
    dh   = np.log(gt_xywh[:, 3] / grid_xywh[:, 3])
    deltas = np.vstack((dxdy, dw, dh)).transpose()
    return deltas


def bbox_transform_inv(grid_xywh, deltas):
    # Assume the input xywh have been normalized.
    new_xywh = grid_xywh.copy()
    new_xywh[:,:2] = new_xywh[:,:2] + np.multiply(deltas[:,:2], grid_xywh[:,2:])
    new_xywh[:,2:] = np.multiply(np.exp(deltas[:,2:]), grid_xywh[:, 2:])
    return new_xywh


def unique_boxes(boxes, scale=1.0):
    # Return indices of unique boxes.
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


def validate_boxes(boxes, width=0, height=0):
    # Check that a set of boxes are valid.
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    assert (x1 >= 0).all()
    assert (y1 >= 0).all()
    assert (x2 >= x1).all()
    assert (y2 >= y1).all()
    assert (x2 < width).all()
    assert (y2 < height).all()


def filter_small_boxes(boxes, min_size):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = np.where((w >= min_size) & (h > min_size))[0]
    return keep


################################################################################
# Box and Images
################################################################################
def create_squared_image(img, pad_value):
    width  = img.shape[1]
    height = img.shape[0]

    max_dim  = np.maximum(width, height)
    offset_x = int(0.5 * (max_dim - width))
    offset_y = int(0.5 * (max_dim - height))

    output_img = pad_value.reshape(1, 1, img.shape[-1]) * \
                 np.ones((max_dim, max_dim, img.shape[-1]))
    output_img[offset_y : offset_y + height, \
               offset_x : offset_x + width,  :] = img

    return output_img, offset_x, offset_y


def crop_image(img, xyxy, pad_value, dilation_ratio=0.0):
    xywh = xyxy_to_xywh(xyxy.reshape((1,4))).flatten()

    factor = 1.0 + dilation_ratio

    img_width  = img.shape[1]
    img_height = img.shape[0]
    out_width  = int(xywh[2] * factor)
    out_height = int(xywh[3] * factor)

    out_img = np.ones((out_height, out_width, 3), dtype=np.float) * \
        pad_value.reshape((1,1,3))

    box_cenx = int(xywh[0])
    box_ceny = int(xywh[1] - 0.5 * xywh[3])
    out_cenx = int(0.5 * out_width)
    out_ceny = int(0.5 * out_height)

    left_radius   = min(box_cenx,              out_cenx)
    right_radius  = min(img_width - box_cenx,  out_cenx)
    top_radius    = min(box_ceny,              out_ceny)
    bottom_radius = min(img_height - box_ceny, out_ceny)

    out_img[(out_ceny-top_radius):(out_ceny+bottom_radius), \
            (out_cenx-left_radius):(out_cenx+right_radius),:] \
            = img[(box_ceny-top_radius):(box_ceny+bottom_radius), \
                  (box_cenx-left_radius):(box_cenx+right_radius),:]

    return out_img


def crop_and_resize(img, xyxy, full_resolution, crop_resolution):
    # xyxy: scaled xyxy
    # full_resolution: output resolution
    # crop_resolution: mask resolution

    width  = xyxy[2] - xyxy[0] + 1
    height = xyxy[3] - xyxy[1] + 1
    cenx = 0.5 * (xyxy[2] + xyxy[0])
    ceny = 0.5 * (xyxy[3] + xyxy[1])
    xywh = np.array([cenx, ceny, width, height])

    img_width  = img.shape[1]
    img_height = img.shape[0]
    out_width  = int(xywh[2] * full_resolution[1]/crop_resolution[1])
    out_height = int(xywh[3] * full_resolution[0]/crop_resolution[0])

    out_img = np.ones((out_height, out_width, 3), dtype=np.float) * \
        cfg.PIXEL_MEANS.reshape((1,1,3))

    box_cenx = int(xywh[0])
    box_ceny = int(xywh[1])
    out_cenx = int(0.5 * out_width)
    out_ceny = int(0.5 * out_height)

    left_radius   = min(box_cenx,              out_cenx)
    right_radius  = min(img_width - box_cenx,  out_cenx)
    top_radius    = min(box_ceny,              out_ceny)
    bottom_radius = min(img_height - box_ceny, out_ceny)

    out_img[(out_ceny-top_radius):(out_ceny+bottom_radius), \
            (out_cenx-left_radius):(out_cenx+right_radius),:] \
            = img[(box_ceny-top_radius):(box_ceny+bottom_radius), \
                  (box_cenx-left_radius):(box_cenx+right_radius),:]

    return cv2.resize(out_img, (full_resolution[1], full_resolution[0])).astype(np.int32)


def dilate_mask(mask, radius):
    inv_mask = 255 - mask
    dm = cv2.distanceTransform(inv_mask, cv2.cv.CV_DIST_L2, 5)
    new_mask = mask.copy()
    new_mask[dm < radius] = 255
    return new_mask


def random_box(full_box, min_dim=50):
    w = full_box[2] - full_box[0] + 1
    h = full_box[3] - full_box[1] + 1

    max_dim = min(w, h)

    if max_dim < min_dim:
        return None

    dim = np.random.randint(min_dim, max_dim+1)

    minx = np.random.randint(full_box[0], full_box[2] - dim + 2)
    miny = np.random.randint(full_box[1], full_box[3] - dim + 2)
    maxx = minx + dim - 1
    maxy = miny + dim - 1

    assert (maxx <= full_box[2] and maxy <= full_box[3])

    return np.array([minx, miny, maxx, maxy])


def random_neighbor_box(full_resolution, cur_box=[0,0,0,0], min_dim=50):
    w = full_resolution[1]
    h = full_resolution[0]

    box_list = []

    box = random_box([0,0,cur_box[0],h-1], min_dim)
    if box is not None:
        box_list = box_list + [box]

    box = random_box([0,0,w-1,cur_box[1]], min_dim)
    if box is not None:
        box_list = box_list + [box]

    box = random_box([0,cur_box[3],w-1,h-1], min_dim)
    if box is not None:
        box_list = box_list + [box]

    box = random_box([cur_box[2],0,w-1,h-1], min_dim)
    if box is not None:
        box_list = box_list + [box]

    num_boxes = len(box_list)
    if num_boxes == 0:
        return None

    index = np.random.randint(0, num_boxes)
    return box_list[index]


def maybe_resize(img, min_edge):
    hh = img.shape[0]; ww = img.shape[1]

    min_dim = min(hh,ww)
    if min_dim <= min_edge:
        return img
    
    if hh < ww:
        scale = float(min_edge)/hh
        ww = int(scale * ww)
        hh = min_edge
        img = cv2.resize(img, (ww, hh))
    else:
        scale = float(min_edge)/ww
        ww = min_edge
        hh = int(scale * hh)
        img = cv2.resize(img, (ww, hh))
    
    return img


def xywh_IoU(A, B):
    I_area = float(min(A[2], B[2]) * min(A[3], B[3]))
    A_area = A[2] * A[3]
    B_area = B[2] * B[3]

    return I_area/(A_area + B_area - I_area)


def xywh_match(A, B):
    A_area = float(A[2] * A[3])
    B_area = float(B[2] * B[3])

    A_ratio = float(A[2])/A[3]
    B_ratio = float(B[2])/B[3]

    a_factor = A_area/B_area

    if a_factor > 2 or a_factor < 0.5 or abs(A_ratio - B_ratio) > 1:
        return 0

    B_h = A[3]
    B_w = B[2] * float(B_h)/B[3]

    return xywh_IoU(A, [B[0], B[1], B_w, B_h])


if __name__ == '__main__':
    xyxy = np.array([ [0, 10, 50, 60], [5, 30, 40, 70.0]])
    xywh = xyxy_to_xywh(xyxy)
    xywh = normalize_xywh(xywh, 200, 200)
    print xywh

    ind  = boxes_to_indices(xywh, [15, 15, 100])
    xywh = indices_to_boxes(ind,  [15, 15, 100])
    print xywh

    xywh = denormalize_xywh(xywh, 200, 200)
    

    new_xyxy = xywh_to_xyxy(xywh, 200, 200)
