import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

############################################################
# Training options
############################################################
__C.TRAIN = edict()
__C.VAL = edict()

# Minibatch size
__C.TRAIN.BATCH_SIZE = 32

# Training epoch
__C.TRAIN.EPOCH = 20

# Clip gradients to avoid exploding gradients
__C.TRAIN.CLIP_GRADIENTS = 5.0

# Initial learning rate
__C.TRAIN.INITIAL_LR = 0.0001

# Regularization
__C.TRAIN.WEIGHT_DECAY = 0.00004

# Dropout rate
__C.TRAIN.DROPOUT = 0.5

# Minimum area of bboxes used for training
__C.TRAIN.MIN_SEG_AREA = 2500

# Area ratio threshold to select background images
__C.TRAIN.MAX_SEG_RATIO = 0.3

# Overlap threshold
__C.TRAIN.OVT = 0.3
__C.VAL.OVT   = 0.2

# Use horizontally-flipped images during training
__C.TRAIN.USE_FLIPPED = False

# Distance threshold to filter objects too close to the edges of the images
__C.TRAIN.DIST_TO_EDGE = 18

# Distance threshold to filter objects too close to the other objects
__C.TRAIN.DIST_TO_OBJ = 18

# To resolve the naming problem of "activation" layer in ResNet50
# __C.TRAIN.LAYER = 'avg_pool'
############################################################

############################################################
# Testing options
############################################################

__C.TEST = edict()

# Aspect ratio threshold to filter the candidates
__C.TEST.ASPECT_RATIO_THRESHOLD = 0.9

# Intersect over union threshold to filter the candidates
__C.TEST.IOU_THRESHOLD = 0.4

# To resolve the naming problem of "activation" layer in ResNet50
# __C.TEST.LAYER = 'avg_pool'

############################################################
# MISC
############################################################

# Resolution of the input image
__C.RESOLUTION = [320, 320, 3]

__C.PREDICT_RESOLUTION = [480, 480, 3]

__C.RETRIEVAL_RESOLUTION = [224, 224, 3]

__C.GRID_SHAPE = [15, 15, 15, 15]

__C.FEAT_DIMS  = [15, 15, 2048]

__C.STATE_DIMS = [15, 15, 512]

__C.PEEPHOLE_RADIUS = 1

# Pixel mean values (BGR order) as a (1, 1, 3) array
# The same pixel mean used in keras (keras/keras/applications/imagenet_utils.py)
__C.PIXEL_MEANS = np.array([[[103.939, 116.779, 123.68]]])

# For reproducibility
__C.RNG_SEED = 0

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

# Layer in ResNet50 that is used for retrieval
__C.LAYER = 'avg_pool'

__C.FILTER_CROWD = False


def get_output_dir(name, net=None):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, name))
    if net is not None:
        outdir = osp.join(outdir, net.name)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    return outdir


def get_imdb_output_dir(imdb, net=None):
    return get_output_dir(imdb.name)


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
