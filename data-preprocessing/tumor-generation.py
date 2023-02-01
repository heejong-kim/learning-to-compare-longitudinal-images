'''
** NOTE **
# OASIS2 dataset can be downloaded here: https://www.oasis-brains.org/#data.
# final demo file and images are provided in: ./data/tumor/images/
'''

import pandas as pd
import numpy as np
import nibabel as nib
import os
import glob
import matplotlib.pyplot as plt
import cv2
import random

def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)

def get_tumor_coordinate(imagemask):
    coordx, coordy = np.where(imagemask > 0)
    medianx, mediany = int((coordx.max() + coordx.min()) / 2), int((coordy.max() + coordy.min()) / 2)
    # add randomness
    if np.random.randint(2):
        medianx += np.random.randint(5, 30)
    else:
        medianx -= np.random.randint(5, 30)

    if np.random.randint(2):
        mediany += np.random.randint(5, 30)
    else:
        mediany -= np.random.randint(5, 30)

    return (medianx, mediany)

def get_biggest_slice_idx(image3d):
    imagemask = image3d>0
    return np.argmax(np.sum(np.sum(imagemask, 0), -1))

set_random_seed(0)



demo = pd.read_csv('/home/hk672/pairwise-comparison/data/OASIS/demo.csv')
imagedir = '/nfs04/data/OASIS-2/preprocessed/'
savedir = './data/tumor/'

t2isoname = imagedir + demo['MRI ID'] + '-mpr-1.nifti.img' #
for i in t2isoname:
    if not os.path.exists(i):
        print(i)

demo['fname'] = t2isoname

# first time point only
demo = demo[demo.Visit == 1].reset_index().drop(columns = ['index', 'Unnamed: 0'])

if not os.path.exists(os.path.join(savedir, 'images')):
    os.makedirs(os.path.join(savedir, 'images'), exist_ok=True)

if not os.path.exists(os.path.join(savedir, 'masks')):
    os.makedirs(os.path.join(savedir, 'masks'), exist_ok=True)

collect_radius = []
collect_demoidx = []
collect_sliceidx = []
collect_fnames = []
collect_subjectid = []


for i in range(len(demo)):
    imgfname = demo.fname.iloc[i]
    subjid = demo['Subject ID'].iloc[i]
    image3d = np.asanyarray(nib.load(imgfname).dataobj)
    if np.sum(image3d) < 100000:
        # TODO: Note that idx 1, 32 used second timepoint
        imgfname = demo.fname.iloc[i].split('MR1-mpr-1.nifti.img')[0] + 'MR2-mpr-1.nifti.img'
        image3d = np.asanyarray(nib.load(imgfname).dataobj)

    sliceidx = get_biggest_slice_idx(image3d)
    image = np.flipud(image3d[:, sliceidx, :])
    cv2.imwrite(os.path.join(savedir+'images/', f'{i}.png'), image/image.max()*255)

    center_coordinates = get_tumor_coordinate((image[:, :, 0] > 0).astype('int'))
    number_of_points = np.random.randint(3, 6, 1)
    init_radius = np.random.permutation(range(3, 21))[0]
    ## linearly growing
    growthratio = 1 + np.random.randint(5,16) / 100
    tumor_color = np.random.randint(230, 251)

    for nop in range(number_of_points[0]):
        collect_demoidx.append(i)
        collect_subjectid.append(subjid)
        if nop == 0:
            current_radius = init_radius
        else:
            current_radius *= growthratio
        collect_radius.append(current_radius)
        collect_sliceidx.append(sliceidx)
        imagemask = (image > 0).astype('int')
        image_rgb = cv2.imread(os.path.join(savedir, f'{i}.png'))
        imageblank = np.zeros(image.shape)
        circle = (cv2.circle(imageblank, (center_coordinates[1],center_coordinates[0]), int(current_radius), (1, 1, 1), -1) > 0).astype('int')
        circle_rgb = np.repeat(circle[:, :, None], 3, 2 )

        for dim in range(3):
            tmpimg = image_rgb[:, :, dim].copy()
            tmpcircle = circle_rgb[:, :, dim].copy()
            tmpimg[tmpcircle > 0] = tumor_color
            image_rgb[:, :, dim] = tmpimg

        # Rotation
        center = (image.shape[1] / 2, image.shape[0] / 2)
        random_rotation_angle = np.random.uniform(-10, 10, 1)[0]
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=random_rotation_angle, scale=1)
        rotated_image = cv2.warpAffine(src=image_rgb, M=rotate_matrix, dsize=(image.shape[1], image.shape[0]))
        rotated_mask = cv2.warpAffine(src=imageblank, M=rotate_matrix, dsize=(image.shape[1], image.shape[0]))

        # Translation
        random_tx, random_ty = np.random.uniform(-10, 10, 1)[0], np.random.uniform(-10, 10, 1)[0]
        random_translation_matrix = np.array([ [1, 0, random_tx],
	                                           [0, 1, random_ty]	], dtype=np.float32)
        translated_image = cv2.warpAffine(src=rotated_image, M=random_translation_matrix, dsize=(image.shape[1], image.shape[0]))
        translated_mask = cv2.warpAffine(src=rotated_mask, M=random_translation_matrix, dsize=(image.shape[1], image.shape[0]))

        cv2.imwrite(os.path.join(savedir+'images', f'{i}-withtumor{nop}.png'), translated_image)
        cv2.imwrite(os.path.join(savedir+'masks', f'{i}-tumormask{nop}.png'), translated_mask * 255)
        collect_fnames.append(os.path.join(savedir+'images', f'{i}-withtumor{nop}.png'))

        del circle;
        del imageblank;
        del imagemask;
        del image_rgb;


newdemo = pd.DataFrame()
newdemo['demoidx'] = collect_demoidx
newdemo['radius'] = collect_radius
newdemo['fname'] = collect_fnames
newdemo['fname'] = newdemo['fname'].str.split(f'{savedir}/images/', expand=True)[1]
newdemo['sliceidx'] = collect_sliceidx
newdemo['Subject ID'] = collect_subjectid

trainvaltest = np.ones(len(newdemo)).astype('str')
for i in range(len(trainvaltest)):
    trainvaltest[i] = np.unique(demo['trainvaltest'][demo['Subject ID'] == newdemo['Subject ID'].iloc[i]])[0]

newdemo['trainvaltest'] = trainvaltest
newdemo['trainvaltest'].value_counts()
newdemo.to_csv(savedir + 'demo-oasis-synthetic-tumor.csv')

