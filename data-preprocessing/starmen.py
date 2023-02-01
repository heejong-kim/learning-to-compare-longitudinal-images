'''
** NOTE **
# Modified starmen dataset can be downloaded from: https://drive.google.com/file/d/16wLtoPTV3ZzOcTbJjA7rYF8sA0oiimGP/view?usp=sharing
# Original starmen dataset can be downloaded from: https://zenodo.org/record/5081988

# This script explains how the modified starmen dataset is generated.
'''


import pandas as pd
import numpy as np
import torch
from PIL import Image
import cv2
import os
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt

# resave images after augmentation
dataset_input_path = './data/'
starmen_df = os.path.join(dataset_input_path, "starmen", "output_random", "df.csv")
df = pd.read_csv(os.path.join(starmen_df), index_col=[0])

newdir = os.path.join(dataset_input_path, "starmen-augmentation", "output_random", "images")
os.makedirs(newdir, exist_ok=True)

augmentation = transforms.Compose([
    transforms.RandomApply(torch.nn.ModuleList(
        [transforms.RandomAffine(translate=(0.1, 0.1), degrees=(-10, 10),
                                 interpolation=InterpolationMode.BILINEAR)]),
        p=0.5), # 0.5
])

for path in df.path:
    img = torch.FloatTensor(np.load(path))
    img_augmented = augmentation(img.unsqueeze(0))
    newfname = path.replace('starmen', 'starmen-augmentation')
    np.save(newfname,img_augmented[0].numpy())

dataset_input_path = './data/'
df.path = df.path.str.replace('starmen', 'starmen-augmentation')
df.to_csv(os.path.join(dataset_input_path, "starmen-augmentation", "output_random", "df.csv"))

starmenaug_df = os.path.join(dataset_input_path, "starmen-augmentation", "output_random", "df.csv")
df = pd.read_csv(os.path.join(starmenaug_df), index_col=[0])

timepoint = np.empty((0)) # 4000 # 1000 # 5000
unqid = np.unique(df.id)
for id in unqid:
    indices = np.array(df[df.id == id].index)
    tsort = np.array(np.argsort(df.t[indices]))
    timepoint = np.concatenate((timepoint,tsort),0)

df['timepoint'] = timepoint
df.path = df.path.str.replace('/home/heejong/HDD4T/projects/pairwise-comparison-longitudinal/'
                              'baseline/longitudinal_autoencoder/data/starmen/'
                              'output_random/','')
df.to_csv(starmen_df)

df.path = df.path.str.replace('/home/heejong/HDD4T/projects/pairwise-comparison-longitudinal/'
                              'baseline/longitudinal_autoencoder/data/starmen-augmentation/'
                              'output_random/','')
df.to_csv(starmenaug_df)
