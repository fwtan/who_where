# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Fuwen Tan @ U.Va (2016)
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import numpy as np
from datasets.coco import coco
# from datasets.old_coco import old_coco
# from datasets.pascal_voc import pascal_voc
# from datasets.visual_genome import visual_genome
# from datasets.cityscapes import cityscapes

# for year in ['2007', '2012']:
#     for split in ['train', 'val', 'trainval', 'test']:
#         name = 'voc_{}_{}'.format(year, split)
#         __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# for year in ['2014']:
#     for split in ['train', 'val', 'minival', 'valminusminival']:
#         name = 'old_coco_{}_{}'.format(year, split)
#         __sets[name] = (lambda split=split, year=year: old_coco(split, year))

# for split in ['all', 'coco', 'yfcc']:
#     name = 'visual_genome_{}'.format(split)
#     __sets[name] = (lambda split=split: visual_genome(split))
#
# # Set up coco_2015_<split>
# for year in ['2015']:
#     for split in ['test', 'test-dev']:
#         name = 'coco_{}_{}'.format(year, split)
#         __sets[name] = (lambda split=split, year=year: coco(split, year))

# # Set up cityscapes_<split>
# for split in ['train', 'val', 'train_extra', 'test']:
#     name = 'cityscapes_{}'.format(split)
#     __sets[name] = (lambda split=split: cityscapes(split))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
