#Convert jpg to grey and output all grey value

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

parser = argparse.ArgumentParser(description='Convert jpg to grey and insert in csv')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
parser.add_argument('--split', type=str, default='PrivateTest', help='split')

opt = parser.parse_args()
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
r1 = range(1,101)
headers = ['emotion','pixels','Usage']
#Firstly read all data from previous csv
with open('/DATA/119/dmchang/srcs/FERPlus/data/fer2013_1.csv','rt')as fin:
    lines=''
    for line in fin:
            lines+=line
fin.close()
#for each pic change to grey and save
for i in r1:
    j = str(i)
    raw_img = io.imread('/DATA/119/dmchang/datas/mooddatabase/negative/male'+j+'.jpg')
    gray = rgb2gray(raw_img)
    gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)
    width,height=gray.shape
    e=[]
    #New a csv file and write datas            
    with open('/DATA/119/dmchang/srcs/FER_py/data/fer2013_2.csv','wt')as fout:       #生成csv文件,有空行
        cout=csv.DictWriter(fout,headers)
        #Preprocess data
        txt=open('/DATA/119/dmchang/srcs/FER_py/data/temp.txt',"w")
            for p in range(height):
                for q in range(width):
                    txt.write(str(gray[q,p]))
                    txt.write(' ')
            txt.close()
        with open('/DATA/119/dmchang/srcs/FER_py/data/temp.txt','r',encoding='utf-8') as f:
            content = f.read()        
            e = ['0',content,'training']
            txt.close()
        cout.writerows(e)
    fout.close()
    
with open('/DATA/119/dmchang/srcs/FER_py/data/fer2013_2.csv','rt')as fin:#读有空行的csv文件，舍弃空行
    for line in fin:
        if line!='\n':
            lines+=line
fin.close()
with open('/DATA/119/dmchang/srcs/FER_py/data/fer2013_2.csv','wt')as fout:  #再次文本方式写入，不含空行
    fout.write(lines)
fout.close()





