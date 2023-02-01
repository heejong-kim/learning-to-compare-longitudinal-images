'''
** NOTE **
# This script explains how the longitudinal images are selected from original embryo dataset.
# "demo.csv" file with selected image information is provided in './data/embryo/'.

## paper: https://www.sciencedirect.com/science/article/pii/S2352340922004607#tbl0001
## dataset: https://zenodo.org/record/6390798#.Y09KKx7MLnE
'''

import os
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


import glob
dir_annotation = '../data/embryo/embryo_dataset_annotations'
embryolist = glob.glob(os.path.join(dir_annotation,'*.csv'))
tmpdemo = pd.DataFrame()
tmpdemo['annotationfiles'] = embryolist
tmp = tmpdemo['annotationfiles'].str.split(dir_annotation+'/',expand=True)
tmp = tmp[1].str.split('_phases.csv', expand=True)
embryonames = np.array(tmp[0])
# phases
phases = np.array(['tPB2', 'tPNa', 'tPNf', \
't2', 't3', 't4', 't5',\
't6', 't7', 't8', 't9+',\
'tM', 'tSB','tB','tEB', 'tHB'])

phase_all = []
phaseidx_all = []
embryonames_all = []
embryoidx_all = []
imagename_all = []
resizefailed = []

for i in range(len(embryonames)):
    embryoname = embryonames[i]
    phasesplit = pd.read_csv(embryolist[i], header=None)
    for j in range(len(phasesplit)):
        phase = phasesplit[0][j]
        phaseidx = np.where(phases == phase)[0][0]
        stidx = int(phasesplit[1][phasesplit[0] == phase])
        endidx = int(phasesplit[2][phasesplit[0] == phase])
        timeframeidx = np.mean([stidx,endidx]).astype('int')

        imgname = f'./data/embryo/embryo_dataset/{embryoname}/F0/*RUN{timeframeidx}.jpeg'
        if not len(glob.glob(imgname))>0:
            imgname = f'./data/embryo/embryo_dataset/{embryoname}/*RUN{timeframeidx}.jpeg'

        if len(glob.glob(imgname))>0:
            imgname = glob.glob(imgname)[0]
            try:
                resize = transforms.Compose([
                    transforms.Resize((200, 200), Image.BICUBIC),
                    transforms.ToTensor(),
                ])
                resizetest = Image.open(imgname)
                resizetest = resize(resizetest)
                resizefailed.append(False)
            except:
                resizefailed.append(True)

            phase_all.append(phase)
            phaseidx_all.append(phaseidx)
            embryonames_all.append(embryoname)
            embryoidx_all.append(i)
            imagename_all.append(imgname)


def find_noncorrupted_image(stidx, endidx,embryoname):
    if stidx == endidx:
        return np.nan, False
    for timeframeidx in range(stidx, endidx):
        imgname = f'./data/embryo/embryo_dataset/{embryoname}/F0/*RUN{timeframeidx}.jpeg'
        if not len(glob.glob(imgname)) > 0:
            imgname = f'./data/embryo/embryo_dataset/{embryoname}/*RUN{timeframeidx}.jpeg'

        if len(glob.glob(imgname)) > 0:
            imgname = glob.glob(imgname)[0]
            try:
                resize = transforms.Compose([
                    transforms.Resize((200, 200), Image.BICUBIC),
                    transforms.ToTensor(),
                ])
                resizetest = Image.open(imgname)
                resizetest = resize(resizetest)
                return timeframeidx, imgname
            except:
                print(imgname)
                return np.nan, False


demo = pd.DataFrame()
demo['embryoname'] = embryonames_all
demo['embryoidx'] = embryoidx_all
demo['phase'] = phase_all
demo['phaseidx'] = phaseidx_all
demo['imagename-fullpath'] = imagename_all
tmp = demo['imagename-fullpath'].str.split('./data/embryo/embryo_dataset/', expand=True)
demo['fname'] = tmp[1]
demo['resizefailed'] = resizefailed

# find noncorrupted image:
for k in range(len(demo)):
    if demo.resizefailed[k]:
        embryoname = demo.embryoname[k]
        i = demo.embryoidx[k]
        phasesplit = pd.read_csv(embryolist[i], header=None)
        phase = demo.phase[k]
        phaseidx = np.where(phases == phase)[0][0]
        stidx = int(phasesplit[1][phasesplit[0] == phase])
        endidx = int(phasesplit[2][phasesplit[0] == phase])
        timeframeidx, newname = find_noncorrupted_image(stidx, endidx, embryoname)
        imagename_all[k] = newname

# TODO: find the changed imagenames and check the file

demonew = demo
del demonew['resizefailed']
demonew['imagename-fullpath'] = imagename_all
# delete failed images
demonew = demonew[demonew['imagename-fullpath'] != False].reset_index().drop(columns = ['index']) # this didn't change # of embryos
tmp = demonew['imagename-fullpath'].str.split('./data/embryo/embryo_dataset/', expand=True)
demonew['fname'] = tmp[1]
demonew.to_csv('./data/embryo/demo.csv')

# train val test embryo-wise
nSubject = len(np.unique(demonew.embryoname))
trainvaltestidx = np.random.permutation(range(nSubject))
trainvaltest = np.zeros(len(trainvaltestidx)).astype('str')
trainvaltest[trainvaltestidx[:int(nSubject*0.6)]] = 'train'
trainvaltest[trainvaltestidx[int(nSubject * 0.6):int(nSubject * 0.8)]] = 'val'
trainvaltest[trainvaltestidx[int(nSubject * 0.8):]] = 'test'

trainvaltestdemo = np.zeros(len(demonew)).astype('str')
for i in range(len(trainvaltest)):
    trainvaltestdemo[demonew.embryoidx == i] = trainvaltest[i]

demonew['trainvaltest'] = trainvaltestdemo
print(f"test:{np.sum(demonew.trainvaltest == 'test')}")
print(f"val:{np.sum(demonew.trainvaltest == 'val')}")
trainset = set(demonew.embryoname[demonew.trainvaltest == 'train'])
testset = set(demonew.embryoname[demonew.trainvaltest == 'test'])
valset = set(demonew.embryoname[demonew.trainvaltest == 'val'])
demonew.to_csv('./data/embryo/demo.csv')

## double check
for x in range(len(demonew)):
    imgname = demonew['imagename-fullpath'].iloc[x]
    try:
        resize = transforms.Compose([
            transforms.Resize((200, 200), Image.BICUBIC),
            transforms.ToTensor(),
        ])
        resizetest = Image.open(imgname)
        resizetest = resize(resizetest)
    except:
        print(demonew.iloc[x])
