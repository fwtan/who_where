#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from datasets.factory import get_imdb
import numpy as np
import cPickle, json, math
import scipy.io as sio
import caffe, os, sys, cv2
import os.path as osp
from glob import glob

def maybe_create(dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

coco_val = get_imdb('coco_2014_val')
CLASSES = coco_val._classes

def filter_detections(dets, thresh):
    cls_boxes   = []
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) > 0:
        for i in inds:
            bb = dets[i, :4].flatten()
            cls_boxes = cls_boxes + [bb]

    return cls_boxes


if __name__ == '__main__':
    import argparse
    np.random.seed(cfg.RNG_SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',  help='input directory of the color images')
    parser.add_argument('--output_dir', help='output directory of the detection files')
    parser.add_argument('--prototxt',   help='path of the prototxt file')
    parser.add_argument('--caffemodel', help='path of the caffe pretrained model')
    opt, unparsed = parser.parse_known_args()

    input_dir  = opt.input_dir
    output_dir = opt.output_dir
    prototxt   = opt.prototxt
    caffemodel = opt.caffemodel
    maybe_create(output_dir)


    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    vis = False
    
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    caffe.set_mode_gpu()
    caffe.set_device(0)
    cfg.GPU_ID = 0
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # input and output dirs

    input_paths = sorted(glob('%s*.jpg'%input_dir))


    for i in range(len(input_paths)):
        im_path = input_paths[i]
        im_name = osp.splitext(osp.basename(im_path))[0]

        timer = Timer()
        timer.tic()
        img = cv2.imread(im_path, cv2.IMREAD_COLOR)
        scores, boxes = im_detect(net, img)

        CONF_THRESH = 0.8
        NMS_THRESH = 0.3

        im_boxes = []
        im_clses = []
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            cur_boxes = filter_detections(dets, thresh=CONF_THRESH)
            if len(cur_boxes) == 0:
                continue
            cur_clses = [cls_ind] * len(cur_boxes)
            im_boxes = im_boxes + cur_boxes
            im_clses = im_clses + cur_clses
            # vis_detections(im, cls, dets, thresh=CONF_THRESH)

        timer.toc()
        print ('{}: {:.3f}s  {:d} objects').format(i, timer.total_time, len(im_clses))

        entry = {}
        entry['image'] = im_name
        entry['boxes'] = np.array(im_boxes).tolist()
        entry['clses'] = im_clses

        json_path = osp.join(output_dir, im_name+'.json')
        with open(json_path, 'w') as res_file:
            json.dump(entry, res_file, indent=4, separators=(',', ': '))

        if vis:
            output_path = osp.join(output_dir, im_name+'.jpg')
            fontScale = 0.0007 * math.sqrt(2 * 512 * 512)

            for j in range(len(im_boxes)):
                bb = im_boxes[j].astype(np.int)
                cls = im_clses[j]
                cls_name = CLASSES[cls]

                cv2.rectangle(img,
                            (bb[0], bb[1]), (bb[2], bb[3]),
                            (0, 255, 0), 1)

                cv2.putText(img,
                            '{:}_{:}'.format(j, cls_name),
                            (bb[0], bb[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 255), 1)
            cv2.imwrite(output_path, img)
