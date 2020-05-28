"""
visualize results for test image
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import transforms as transforms
starttime = time.time()
from torch.autograd import Variable
from fer import FER2013

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *
import utils
from models import vgg_prune

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

checkpoint1 = torch.load('models/prune_models/PrivateTest_modelprune.t7',map_location='cpu')
net = vgg_prune.VGG(cfg=checkpoint1['cfg'])
checkpoint2 = torch.load('models/fer_Pri_vgg19_private.t7',map_location='cpu')
net.load_state_dict(checkpoint2['state_dict'])
endtime = time.time()
dtime_1 = endtime - starttime
#checkpoint = torch.load('models/fer_Pri_vgg19_private.t7')
#net = vgg_prune.VGG(cfg=checkpoint['cfg'])
#net.load_state_dict(checkpoint['state_dict'])

#net.cuda()
#net.eval()

criterion = nn.CrossEntropyLoss()

#private test
#PrivateTestset = FER2013(split = 'PrivateTest', transform=transform_test)
#PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=16, shuffle=False, num_workers=1)

#def test():
    
#    net.eval()
    #bin_op.binarization()
#    PrivateTest_loss = 0
#    correct = 0
#    total = 0
#    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
#        bs, ncrops, c, h, w = np.shape(inputs)
#        inputs = inputs.view(-1, c, h, w)        
#        inputs, targets = inputs.cuda(), targets.cuda()
#        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
#        outputs = net(inputs)
#        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
#        loss = criterion(outputs_avg, targets)
#        PrivateTest_loss += loss.item()
#        _, predicted = torch.max(outputs_avg.data, 1)
#        total += targets.size(0)
#        correct += predicted.eq(targets.data).cpu().sum()
#        utils.progress_bar(batch_idx, len(PrivateTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (PrivateTest_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))
        #bin_op.restore()
#    return
#test()

r1 = range(1,120)
for i in r1:
    j = str(i)
    raw_img = io.imread('images/'+j+'.jpg')
    gray = rgb2gray(raw_img)
    gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)
    img = gray[:, :, np.newaxis]
    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)
    ncrops, c, h, w = np.shape(inputs)
    inputs = inputs.view(-1, c, h, w)
    #inputs = inputs.cuda()
    #with torch.no_grad():
    #    inputs = Variable(inputs)
    #inputs = Variable(inputs, volatile=True)
    outputs = net(inputs)

    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

    score = F.softmax(outputs_avg,dim=0)
    _, predicted = torch.max(outputs_avg.data, 0)
    plt.rcParams['figure.figsize'] = (13.5,5.5)
    axes=plt.subplot(1, 3, 1)
    plt.imshow(raw_img)
    plt.xlabel('Input Image', fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)
    plt.subplot(1, 3, 2)
    ind = 0.1+0.6*np.arange(len(class_names))    # the x locations for the groups
    width = 0.4       # the width of the bars: can also be len(x) sequence
    color_list = ['red','orangered','darkorange','limegreen','darkgreen','royalblue','navy']
    print('No'+j)
    for k in range(len(class_names)):
       plt.bar(ind[k], score.data.cpu().numpy()[k], width, color=color_list[k])
       print(class_names[k],':',score.data.cpu().numpy()[k])
    plt.title("Classification results ",fontsize=20)
    plt.xlabel(" Expression Category ",fontsize=16)
    plt.ylabel(" Classification Score ",fontsize=16)
    plt.xticks(ind, class_names, rotation=45, fontsize=14)
    axes=plt.subplot(1, 3, 3)
    # show emojis
    emojis_img = io.imread('images/emojis/%s.png' % str(class_names[int(predicted.cpu().numpy())]))
    plt.imshow(emojis_img)
    plt.xlabel('Emoji Expression', fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout()
    
    print('Saving Images')
    #plt.show()
    plt.savefig('/DATA/119/dmchang/srcs/FER_py/images/results/'+j+'.png')
    plt.close()
    print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))

endtime = time.time()
dtime = endtime - starttime
print("Processing time: %s" % dtime)
print("load time: %s" % dtime_1)
#print("The Score is %d" %str(class_names[int(predicted.cpu().numpy())]))


