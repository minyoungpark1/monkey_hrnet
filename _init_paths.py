# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

if __name__== "__main__":
    #this_dir = osp.dirname("__file__")
    this_dir = os.getcwd()

    lib_path = os.path.join(this_dir, '..', 'lib')
    add_path(lib_path)

    mm_path = os.path.join(this_dir, '..', 'lib/poseeval/py-motmetrics')
    add_path(mm_path)
