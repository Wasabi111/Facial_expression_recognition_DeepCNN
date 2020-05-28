#Convert jpg/png/gif to grey and output all grey value to csv

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import csv
import torch.nn as nn
import torch.nn.functional as F
import os
import transforms as transforms
from skimage import io
from skimage.transform import resize
import cv2

#parser = argparse.ArgumentParser(description='Convert jpg to grey and insert in csv')
#parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
#parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
#parser.add_argument('--split', type=str, default='PrivateTest', help='split')
#opt = parser.parse_args()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def iter_frames(im):
    try:
        i= 0
        while 1:
            im.seek(i)
            imframe = im.copy()
            if i == 0:
                palette = imframe.getpalette()
            else:
                imframe.putpalette(palette)
            yield imframe
            i += 1
    except EOFError:
        pass

#class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

headers = ['emotion','pixels','Usage']
#Firstly read all data from previous csv
with open('/DATA/119/dmchang/srcs/FER_py/data/fer2013_1.csv','rt')as fin:
    lines=''
    for line in fin:
            lines+=line
fin.close()
#for each pic change to grey and save
r1 = range(1,250)
with open('/DATA/119/dmchang/srcs/FER_py/data/fer2013_2.csv','wt')as fout:
    for i in r1:
        j = str(i)
        
        #gif
        #im = Image.open('/DATA/119/dmchang/datas/CK+/positive/female/'+j+'.gif')
        #for t, frame in enumerate(iter_frames(im)):
        #    frame.save('/DATA/119/dmchang/datas/image.png',**frame.info)
        
        #raw_img = Image.open('/DATA/119/dmchang/datas/image.png')
        raw_img = Image.open('/DATA/119/dmchang/datas/CK+/surprise/5 ('+j+').png')
        gray = raw_img.convert('L')
        gray.save('/DATA/119/dmchang/datas/image_1.png')
        raw_img = io.imread('/DATA/119/dmchang/datas/image_1.png')
        #gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)
        gray = resize(raw_img, (48,48), mode='symmetric')
        width,height=gray.shape
        e=[]
        #Preprocess data
        os.remove('/DATA/119/dmchang/srcs/FER_py/data/temp.txt')
        txt=open('/DATA/119/dmchang/srcs/FER_py/data/temp.txt',"w")
        for p in range(height):
            for q in range(width):
                txt.write(str(int(gray[p,q]*255)))
                txt.write(' ')
        txt.close()
        with open('/DATA/119/dmchang/srcs/FER_py/data/temp.txt','r') as f:
            content = f.read()        
        f.close()
    #New a csv file and write datas            
        #cout=csv.DictWriter(fout,headers)
        #cout.writeheader()
        fout.write('5'+',')
        fout.write(content+',')
        fout.write('training'+'\n')
fout.close()

with open('/DATA/119/dmchang/srcs/FER_py/data/fer2013_2.csv','rt')as fin:
    for line in fin:
        lines+=line
fin.close()
with open('/DATA/119/dmchang/srcs/FER_py/data/fer2013_1.csv','wt')as fout:
    fout.write(lines)
fout.close()


print('converted successfully')





