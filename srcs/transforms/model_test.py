import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import transforms as transforms

from torch.autograd import Variable
from fer import FER2013

import transforms as transforms

from models import *
import utils
from models import vgg_prune
starttime = time.time()

checkpoint1 = torch.load('models/prune_models/fer_Pri_vgg16_prune.pth')
net = vgg_prune.VGG(cfg=checkpoint1['cfg'])
checkpoint2 = torch.load('models/prune_models/fer_Pri_vgg16_refine.pth')
net.load_state_dict(checkpoint2['state_dict'])

#checkpoint = torch.load('models/prune_models/PrivateTest_modelprune.t7')
#checkpoint = torch.load('models/prune_models/PrivateTest_modelprune.t7')
#net = vgg_prune.VGG(cfg=checkpoint['cfg'])
#net.load_state_dict(checkpoint['state_dict'])

midtime = time.time()
dtime_1 = midtime - starttime
print("Preprocessing time: %s" % dtime_1)


print(cfg)
print(net)
