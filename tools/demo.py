#!/usr/bin/env python

import _init_paths
import os, sys
import cv2, json
import math
from time import time
import numpy as np
import os.path as osp
import datasets.imdb
from datasets.factory import get_imdb
import datasets.ds_utils as ds_utils
from config import cfg, get_output_dir
from html import HTML
from glob import glob
from models.PredictionCNN import PredictionCNN
from models.RetrievalCNN import RetrievalCNN


def create_testdb():
    colors_dir  = osp.abspath('../data/testset/test_colors')
    layouts_dir = osp.abspath('../data/testset/test_layouts')

    color_paths = sorted(glob('%s/*.jpg'%colors_dir))
    names = [osp.splitext(osp.basename(x))[0] for x in color_paths]

    testdb = []
    for i in range(len(names)):
        entry = {}
        name  = names[i]
        img_path = osp.join(colors_dir, name+'.jpg')
        lyo_path = osp.join(layouts_dir, name+'.jpg')
            
        entry['name']  = name
        entry['bg_image']  = img_path
        entry['bg_layout'] = lyo_path
            
        testdb = testdb + [entry]

    return testdb


if __name__ == '__main__':
    np.random.seed(cfg.RNG_SEED)
    val_imdb = get_imdb('coco_2014_val')
    testdb = create_testdb()

    predictor = PredictionCNN('../output/')
    retriever = RetrievalCNN('../output/')
    assert(predictor.load_checkpoint('../data/pretrained/prediction_ckpts'))
    start = time()
    print("Composite image generation ..")
    testdb = predictor.sampler(testdb, K=1, vis=True)
    retriever.sample(testdb, val_imdb.objdb, mode=0, K=1, show_gt=False)
    print('Took {:.3f}s for {:d} test images').format(time()-start, len(testdb))
    print('Results have been written to "../output/composite_colors"')

    




    
    
    

