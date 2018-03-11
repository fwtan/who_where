# --------------------------------------------------------
# Who_where
# Copyright (c) 2016 University of Virginia
# Licensed under The MIT License [see LICENSE for details]
# Written by Fuwen Tan @ U.Va (2017)
# --------------------------------------------------------


import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add keras to PYTHONPATH
keras_path = osp.join(this_dir, '..', 'keras')
add_path(keras_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)
