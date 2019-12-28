from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from time import time
import misc.utils as utils
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet
import misc.utils as utils

import sys
sys.path.append("/home1/06008/xf993/self-critical.pytorch/cider")
sys.path.append("/home/ziyu/self-critical.pytorch/cider")
from pyciderevalcap.ciderD.ciderD import CiderD
sys.path.append("/home1/06008/xf993/self-critical.pytorch/coco-caption")
sys.path.append("/home/ziyu/self-critical.pytorch/coco-caption")
from pycocoevalcap.bleu.bleu import Bleu