#!/usr/bin/env python

import _init_paths
import os, sys
import cv2, json
import math, time
import numpy as np
import os.path as osp
import datasets.imdb
from datasets.factory import get_imdb
import datasets.ds_utils as ds_utils
from config import cfg, get_output_dir
from glob import glob


def render_layout(name, color_palette):
    img_path = osp.join(imgs_dir, name+'.jpg')
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    width = img.shape[1]
    height = img.shape[0]

    det_path = osp.join(dets_dir, name+'.json')
    det_info = json.loads(open(det_path,'r').read())

    boxes = np.array(det_info['boxes']).copy().reshape((-1, 4)).astype(np.int)
    clses = np.array(det_info['clses']).flatten()


    output = ds_utils.scene_layout(width, height, boxes, clses, color_palette)

    return output

if __name__ == '__main__':
    import argparse
    np.random.seed(cfg.RNG_SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_images_dir',     help='directory of the input color images')
    parser.add_argument('--input_detections_dir', help='directory of the input detection files')
    parser.add_argument('--output_layouts_dir',   help='directory of the output layout images')
    opt, unparsed = parser.parse_known_args()

    imgs_dir = opt.input_images_dir
    dets_dir = opt.input_detections_dir
    layouts_dir = opt.output_layouts_dir
    ds_utils.maybe_create(layouts_dir)

    palette_path = osp.join(cfg.DATA_DIR, 'coco', 'color_palette.json')
    color_palette = json.loads(open(palette_path,'r').read())
    img_paths = sorted(glob(osp.join(imgs_dir, '*')))
    img_names = [osp.splitext(osp.basename(x))[0] for x in img_paths]

    for i in range(len(img_names)):
        x = img_names[i]
        output = render_layout(x, color_palette)
        output_path = osp.join(layouts_dir, x+'.jpg')
        cv2.imwrite(output_path, output)
        print i
    
    
