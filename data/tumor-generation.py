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
    # coordidx = np.random.permutation(range(len(coordx)))[0]
    # medianx, mediany = np.median(coordx).astype('int'), np.median(coordy).astype('int')
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

def get_two_tumor_coordinate(imagemask):
    coordx, coordy = np.where(imagemask > 0)
    # coordidx = np.random.permutation(range(len(coordx)))[0]
    # medianx, mediany = np.median(coordx).astype('int'), np.median(coordy).astype('int')
    medianx, mediany = int((coordx.max() + coordx.min()) / 2), int((coordy.max() + coordy.min()) / 2)

    def get_xy(medianx, mediany):
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

    medianx1, mediany1 = get_xy(medianx, mediany)
    medianx2, mediany2 = get_xy(medianx, mediany)

    while np.square(medianx1-medianx2) + np.square(mediany1-mediany2) < 400:
        medianx1, mediany1 = get_xy(medianx, mediany)
        medianx2, mediany2 = get_xy(medianx, mediany)

    return (medianx1, mediany1), (medianx2, mediany2)

def get_biggest_slice_idx(image3d):
    imagemask = image3d>0
    return np.argmax(np.sum(np.sum(imagemask, 0), -1))

set_random_seed(0)

demo = pd.read_csv('/home/hk672/pairwise-comparison/data/OASIS/demo.csv')
## preprocessed images
t2isoname = '/nfs04/data/OASIS-2/preprocessed/' +   demo['MRI ID'] + '-mpr-1.nifti.img' #
for i in t2isoname:
    if not os.path.exists(i):
        print(i)

demo['fname'] = t2isoname

# first time point only
demo = demo[demo.Visit == 1].reset_index().drop(columns = ['index', 'Unnamed: 0'])

# TODO: ValueError: zero-size array to reduction operation maximum which has no identity
savedir = '/home/hk672/pairwise-comparison-longitudinal/data/oasis-tumor/'
# savedir = '/home/hk672/pairwise-comparison-longitudinal/data/oasis-tumor-wo-preprocess/'

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
        # cv2.imwrite(os.path.join(savedir, f'{i}-tumormask{nop}.png'), circle / circle.max() * 255)
        # cv2.imwrite(os.path.join(savedir, f'{i}-withtumor{nop}.png'), image_rgb)
        # cv2.imwrite(os.path.join(savedir, f'{i}-withtumor{nop}-rotation.png'), rotated_image)
        # cv2.imwrite(os.path.join(savedir, f'{i}-withtumor{nop}-rotation-translation.png'), translated_image)
        collect_fnames.append(os.path.join(savedir+'images', f'{i}-withtumor{nop}.png'))

        del circle;
        del imageblank;
        del imagemask;
        del image_rgb;



newdemo = pd.DataFrame()
newdemo['demoidx'] = collect_demoidx
newdemo['radius'] = collect_radius
newdemo['fname'] = collect_fnames
# file name only w/o location
# newdemo['fname'] = newdemo['fname'].str.split('/home/hk672/pairwise-comparison-longitudinal/data/oasis-tumor/images/', expand=True)[1]
newdemo['fname'] = newdemo['fname'].str.split(f'{savedir}/images/', expand=True)[1]
newdemo['sliceidx'] = collect_sliceidx
newdemo['Subject ID'] = collect_subjectid

trainvaltest = np.ones(len(newdemo)).astype('str')
for i in range(len(trainvaltest)):
    trainvaltest[i] = np.unique(demo['trainvaltest'][demo['Subject ID'] == newdemo['Subject ID'].iloc[i]])[0]

newdemo['trainvaltest'] = trainvaltest
newdemo['trainvaltest'].value_counts()
newdemo.to_csv(savedir + 'demo-oasis-synthetic-tumor.csv')

#############################################
## ------- tumors without preprocessing
# get mid slice from previous dataset
demoref = pd.read_csv('/home/hk672/pairwise-comparison-longitudinal/data/oasis-tumor/' + 'demo-oasis-synthetic-tumor.csv')
demo = pd.read_csv('/home/hk672/pairwise-comparison/data/OASIS/demo.csv')
## not preprocessed
t2isoname = '/nfs04/data/OASIS-2/OAS2_RAW/' +   demo['MRI ID'] + '/RAW/mpr-1.nifti.img' #
for i in t2isoname:
    if not os.path.exists(i):
        print(i)

demo['fname'] = t2isoname
# first time point only
demo = demo[demo.Visit == 1].reset_index().drop(columns = ['index', 'Unnamed: 0'])

# TODO: ValueError: zero-size array to reduction operation maximum which has no identity
savedir = '/home/hk672/pairwise-comparison-longitudinal/data/oasis-tumor-wo-preprocess/'

if not os.path.exists(os.path.join(savedir, 'images')):
    os.makedirs(os.path.join(savedir, 'images'), exist_ok=True)

if not os.path.exists(os.path.join(savedir, 'masks')):
    os.makedirs(os.path.join(savedir, 'masks'), exist_ok=True)

# used preprocessed demo and same parameter for tumor size

# TODO collcet colors
meancollect = []
maxcollect = []
for i in range(len(demo)):
    image_rgb = cv2.imread(os.path.join(savedir, 'images', f'{i}.png'))
    meancollect.append(np.mean(image_rgb))
    maxcollect.append(np.max(image_rgb))

collect_radius = []
collect_demoidx = []
collect_sliceidx = []
collect_fnames = []
collect_subjectid = []
collect_tumorcolor = []
for i in range(len(demo)):
    imgfname = demo.fname.iloc[i]
    subjid = demo['Subject ID'].iloc[i]
    image3d = np.asanyarray(nib.load(imgfname).dataobj)
    if np.sum(image3d) < 100000:
        # TODO: Note that idx 1, 32 used second timepoint
        imgfname = demo.fname.iloc[i].split('MR1-mpr-1.nifti.img')[0] + 'MR2-mpr-1.nifti.img'
        image3d = np.asanyarray(nib.load(imgfname).dataobj)

    assert np.all(demoref['sliceidx'][demoref['Subject ID'] == subjid]), "all slices from the same image should be the same"
    sliceidx = np.unique(demoref['sliceidx'][demoref['Subject ID'] == subjid])[0]
    image = np.flipud(image3d[:, sliceidx, :])
    cv2.imwrite(os.path.join(savedir+'images/', f'{i}.png'), image/image.max()*255)

    center_coordinates = get_tumor_coordinate((image[:, :, 0] > 0).astype('int'))
    tumor_color = np.random.randint(151, 201)
    # tumor_color = np.random.randint(image.mean(), image.mean()+(0.5*image.std()))

    number_of_points = np.sum(demoref['Subject ID'] == subjid)
    demoref_subj = demoref[demoref['Subject ID'] == subjid].reset_index()

    for nop in range(len(demoref_subj)):
        collect_demoidx.append(i)
        collect_subjectid.append(subjid)
        current_radius = demoref_subj['radius'].iloc[nop]
        collect_radius.append(current_radius)
        collect_sliceidx.append(sliceidx)
        collect_tumorcolor.append(tumor_color)

        imagemask = (image > 0).astype('int')
        image_rgb = cv2.imread(os.path.join(savedir, 'images', f'{i}.png'))
        imageblank = np.zeros(image.shape)
        circle = (cv2.circle(imageblank, (center_coordinates[1],center_coordinates[0]), int(current_radius), (1, 1, 1), -1) > 0).astype('int')
        circle_rgb = np.repeat(circle[:, :, None], 3, 2 )

        for dim in range(3):
            tmpimg = image_rgb[:, :, dim].copy()
            tmpcircle = circle_rgb[:, :, dim].squeeze().copy()
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

        collect_fnames.append(f'{i}-withtumor{nop}.png') #  os.path.join(savedir+'images', f'{i}-withtumor{nop}.png')

        del circle;
        del imageblank;
        del imagemask;
        del image_rgb;



newdemo = pd.DataFrame()
newdemo['demoidx'] = collect_demoidx
newdemo['radius'] = collect_radius
newdemo['fname'] = collect_fnames
newdemo['sliceidx'] = collect_sliceidx
newdemo['Subject ID'] = collect_subjectid
newdemo['color'] = collect_tumorcolor

trainvaltest = np.ones(len(newdemo)).astype('str')
for i in range(len(trainvaltest)):
    trainvaltest[i] = np.unique(demo['trainvaltest'][demo['Subject ID'] == newdemo['Subject ID'].iloc[i]])[0]

newdemo['trainvaltest'] = trainvaltest
newdemo['trainvaltest'].value_counts()
newdemo.to_csv(savedir + 'demo-oasis-synthetic-tumor.csv')


#############################################
## ------- extra: Two tumors
savedir = '/home/hk672/pairwise-comparison-longitudinal/data/oasis-tumors/'

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

    center_coordinates1, center_coordinates2 = get_two_tumor_coordinate((image > 0).astype('int'))

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
        image_rgb = cv2.imread(os.path.join(savedir, 'images', f'{i}.png'))
        imageblank = np.zeros(image.shape)
        circle1 = (cv2.circle(imageblank, (center_coordinates1[1],center_coordinates1[0]), int(current_radius), (1, 1, 1), -1) > 0).astype('int')
        circle_rgb1 = np.repeat(circle1[:, :, None], 3, 2 )
        circle2 = (cv2.circle(imageblank, (center_coordinates2[1],center_coordinates2[0]), int(current_radius), (1, 1, 1), -1) > 0).astype('int')
        circle_rgb2 = np.repeat(circle2[:, :, None], 3, 2 )

        for dim in range(3):
            tmpimg = image_rgb[:, :, dim].copy()
            tmpcircle = circle_rgb1[:, :, dim].copy()
            tmpimg[tmpcircle > 0] = tumor_color
            tmpcircle = circle_rgb2[:, :, dim].copy()
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
        # cv2.imwrite(os.path.join(savedir, f'{i}-tumormask{nop}.png'), circle / circle.max() * 255)
        # cv2.imwrite(os.path.join(savedir, f'{i}-withtumor{nop}.png'), image_rgb)
        # cv2.imwrite(os.path.join(savedir, f'{i}-withtumor{nop}-rotation.png'), rotated_image)
        # cv2.imwrite(os.path.join(savedir, f'{i}-withtumor{nop}-rotation-translation.png'), translated_image)
        collect_fnames.append(os.path.join(savedir+'images', f'{i}-withtumor{nop}.png'))

        del circle1;
        del circle2;
        del imageblank;
        del imagemask;
        del image_rgb;


newdemo = pd.DataFrame()
newdemo['demoidx'] = collect_demoidx
newdemo['radius'] = collect_radius
newdemo['fname'] = collect_fnames
# file name only w/o location
newdemo['fname'] = newdemo['fname'].str.split('/home/hk672/pairwise-comparison-longitudinal/'
                                              'data/oasis-tumors/images/', expand=True)[1]
newdemo['sliceidx'] = collect_sliceidx
newdemo['Subject ID'] = collect_subjectid

trainvaltest = np.ones(len(newdemo)).astype('str')
for i in range(len(trainvaltest)):
    trainvaltest[i] = np.unique(demo['trainvaltest'][demo['Subject ID'] == newdemo['Subject ID'].iloc[i]])[0]

newdemo['trainvaltest'] = trainvaltest
newdemo['trainvaltest'].value_counts()
newdemo['timepoint'] = newdemo.fname.str.split('.png', expand=True)[0].str[-1].astype('int')
newdemo.to_csv(savedir + 'demo-oasis-synthetic-tumor.csv')



## -- additional rotation / translation test
import torch
savedir = '/home/hk672/pairwise-comparison-longitudinal/data/oasis-tumor/'
dfadd = pd.read_csv(savedir + 'demo-oasis-synthetic-tumor.csv')
dfaddtest = dfadd[dfadd.trainvaltest == 'test'].reset_index().drop(columns= ['Unnamed: 0'])
selectedid = np.unique(dfaddtest['Subject ID'])[-1]
dfaug = dfadd[dfadd['Subject ID'] == selectedid].reset_index().drop(columns = ['Unnamed: 0'])

baseimage = f'{savedir}/images/40.png'
transimage = f'{savedir}/images/40.png'

newsavedir = '/home/hk672/pairwise-comparison-longitudinal/data/oasis-tumor-additional/images'

fname = []
fname.append(baseimage)
fname.append(transimage)
addrotate = [ 10, 20, 30, 40]
addtranslate = [ -20, -10, 0, 10, 20]


collect_radius = []
collect_demoidx = []
collect_fnames = []
collect_subjectid = []

i = 40
subjid = dfadd['Subject ID'].iloc[i]
image = cv2.imread(os.path.join(savedir+'images/', f'{i}.png'))
center_coordinates = get_tumor_coordinate((image[:, :, 0] > 0).astype('int'))

# number_of_points = np.random.randint(3, 6, 1)
init_radius = 10
growthratio = 1 + .1
tumor_color = 240
current_radius = init_radius * growthratio

for rot in addrotate:
    trans = 0
    collect_demoidx.append(i)
    collect_subjectid.append(subjid)
    collect_radius.append(current_radius)
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
    # random_rotation_angle = np.random.uniform(-10, 10, 1)[0]
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rot, scale=1)
    rotated_image = cv2.warpAffine(src=image_rgb, M=rotate_matrix, dsize=(image.shape[1], image.shape[0]))
    fname = os.path.join(newsavedir, f'{i}-withtumor-rot{rot}-trans{trans}.png')
    cv2.imwrite(fname, translated_image)
    collect_fnames.append(fname)

    del circle;
    del imageblank;
    del imagemask;
    del image_rgb;


for rot in addrotate:
    trans = 0
    collect_demoidx.append(i)
    collect_subjectid.append(subjid)
    collect_radius.append(current_radius)
    image_rgb = cv2.imread(os.path.join(savedir, f'{i}.png'))
    imageblank = np.zeros(image.shape)
    circle = (cv2.circle(imageblank, (center_coordinates[1],center_coordinates[0]), int(current_radius), (1, 1, 1), -1) > 0).astype('int')
    circle_rgb = np.repeat(circle[:, :, None], 3, 2 )

    for dim in range(3):
        tmpimg = image_rgb[:, :, dim].copy()
        tmpcircle = circle_rgb[:, :, dim].copy()
        tmpimg[tmpcircle > 0] = tumor_color
        image_rgb[:, :, dim] = tmpimg


    # Translation
    random_tx, random_ty = np.random.uniform(-10, 10, 1)[0], np.random.uniform(-10, 10, 1)[0]
    random_translation_matrix = np.array([ [1, 0, random_tx],
                                           [0, 1, random_ty]	], dtype=np.float32)
    translated_image = cv2.warpAffine(src=rotated_image, M=random_translation_matrix, dsize=(image.shape[1], image.shape[0]))
    fname = os.path.join(newsavedir, f'{i}-withtumor-rot{rot}-trans{trans}.png')
    cv2.imwrite(fname, translated_image)
    collect_fnames.append(fname)

    del circle;
    del imageblank;
    del imagemask;
    del image_rgb;


newdemo = pd.DataFrame()
newdemo['demoidx'] = collect_demoidx
newdemo['radius'] = collect_radius
newdemo['fname'] = collect_fnames
# file name only w/o location
newdemo['fname'] = newdemo['fname'].str.split('/home/hk672/pairwise-comparison-longitudinal/data/oasis-tumor/images/', expand=True)[1]
newdemo['sliceidx'] = collect_sliceidx
newdemo['Subject ID'] = collect_subjectid

trainvaltest = np.ones(len(newdemo)).astype('str')
for i in range(len(trainvaltest)):
    trainvaltest[i] = np.unique(demo['trainvaltest'][demo['Subject ID'] == newdemo['Subject ID'].iloc[i]])[0]

newdemo['trainvaltest'] = trainvaltest
newdemo['trainvaltest'].value_counts()