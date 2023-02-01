'''
** NOTE **
# preprocessed slice OASIS images can be found here: https://drive.google.com/file/d/1FDW8t0DoaQ2xvqul9sXU2BbqOvHr_Zod/view?usp=share_link

# OASIS3 original dataset can be downloaded from here: https://www.oasis-brains.org/#data.
# This script explains how the OASIS3 dataset is preprocessed.
# The process includes: 1) download images 2) image registration (affine) 3) create demo file (demo-healthy-longitudinal.csv).
'''

import pandas as pd
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
import glob
from PIL import Image

mriinfo = pd.read_csv('/nfs04/data/OASIS3/demo/mri-info.csv')
subjectinfo = pd.read_csv('/nfs04/data/OASIS3/demo/subject-info.csv')

# MR count more than 1
subjectinfo = subjectinfo[subjectinfo['MR Count'] > 1.0].reset_index().drop(columns=['index'])
subjectinfo = subjectinfo.drop(index=[0]).reset_index().drop(columns=['index'])

# Scanner 3T
mriinfo = mriinfo[mriinfo.Scanner.astype('str') == '3.0T'].reset_index().drop(columns=['index'])

## https://central.xnat.org/app/action/DisplayItemAction/search_element/xnat%3AmrSessionData/search_field/xnat%3AmrSessionData.ID/search_value/CENTRAL04_E04187/popup/false/project/OASIS3
# data demographics
demo = pd.read_csv('/nfs04/data/OASIS3/demo/OASIS3_data_files/scans/demo-demographics/resources/csv/files/OASIS3_demographics.csv')

# create csv for scan download list
np.savetxt('/nfs04/data/OASIS3/demo/oasis/oasis3-mr-list.csv', np.array(mriinfo.Label), '%s')

# bash script
demodir = '/nfs04/data/OASIS3/demo'
scriptname = '/home/hk672/oasis-data/download_scans/download_oasis_scans_bids.sh'
inputfile = os.path.join(demodir, f'oasis3-mr-all.csv')
savedir = '/nfs04/data/OASIS3/image'
username = 'heejong'
scantype = 'T1w'
os.makedirs(savedir, exist_ok=True)
os.system(f'{scriptname} {inputfile} {username} {scantype}')

# parallelize
imagelist = np.loadtxt(inputfile, str)
num_cores = 400
batch = int(len(imagelist) / num_cores)

for i in range(batch + 1):
    print(i*num_cores, (i*num_cores)+ num_cores)
    print(imagelist[i*num_cores:(i*num_cores) + num_cores])
    np.savetxt(inputfile.split('.csv')[0] + str(i) + '.csv', imagelist[i * num_cores:(i * num_cores) + num_cores], '%s')

def run_bash_parallel(i):
    demodir = '/nfs04/data/OASIS3/demo'
    scriptname = '/home/hk672/oasis-data/download_scans/download_oasis_scans_bids.sh'
    inputfile = os.path.join(demodir, f'oasis3-mr-list{i}.csv')
    savedir = '/nfs04/data/OASIS3/image'
    scantype = 'T1w'
    os.system(f'{scriptname} {inputfile} {savedir} {scantype}')
    return i

from joblib import Parallel, delayed
out = Parallel(n_jobs=num_cores)(delayed(run_bash_parallel)(i) for i in range(batch + 1))

# how many do we have
all_nifti_names = glob.glob('/nfs04/data/OASIS3/image/sub-*/*/*/*.nii.gz')
demo = pd.DataFrame({})
demo['fname'] = all_nifti_names
tmp = demo.fname.str.split('/nfs04/data/OASIS3/image/', expand=True)
sid = pd.DataFrame(tmp[1].str.split('/', expand=True)[0])[0].str.split('-', expand=True)[1]
sessid = tmp[1].str.split('/', expand=True)[1]
demo['subject-id'] = sid
demo['session-id'] = sessid

# match demographics
demo_demo = pd.read_csv('/nfs04/data/OASIS3/demo/OASIS3_data_files/scans/'
                        'demo-demographics/resources/csv/files/'
                        'OASIS3_demographics.csv')
demo_cognitive = pd.read_csv('/nfs04/data/OASIS3/demo/OASIS3_data_files/scans/'
                             'pychometrics-Form_C1__Cognitive_Assessments/'
                             'resources/csv/files/OASIS3_UDSc1_cognitive_assessments.csv')

demo_diagnosis = pd.read_csv('/nfs04/data/OASIS3/demo/OASIS3_data_files/scans/'
                             'UDSd1-Form_D1__Clinician_Diagnosis___Cognitive_Status_and_Dementia/'
                             'resources/csv/files/OASIS3_UDSd1_diagnoses.csv')

demo_dic = pd.read_csv('/nfs04/data/OASIS3/demo/OASIS3_data_files/scans/'
                       'dictionaries-Imaging_and_UDS_data_dictionaries/'
                       'resources/csv/files/')

demo_cdr = pd.read_csv('/nfs04/data/OASIS3/demo/OASIS3_data_files/scans/'
                       'UDSb4-Form_B4__Global_Staging__CDR__Standard_and_Supplemental/'
                       'resources/csv/files/OASIS3_UDSb4_cdr.csv')
naming = demo_cdr['OASIS_session_label'].str.split('_', expand=True)
naming_subject = np.array(naming[0])
naming_session = np.array('ses-'+naming[2])

demo_mri = pd.read_csv('/nfs04/data/OASIS3/demo/OASIS3_data_files/scans/'
                       'MRI-json-MRI_json_information/resources/csv/files/OASIS3_MR_json.csv')

cdr_summary = demo_cdr.groupby('OASISID')['CDRSUM'].sum()
cdr0_subject_list = cdr_summary[cdr_summary == 0].index

HCsubjects = np.unique(demo['subject-id'][demo['subject-id'].isin(cdr0_subject_list)])
demoHC = demo[demo['subject-id'].isin(HCsubjects)].reset_index().drop(columns = ['index'])
demoHC['id-session'] = demoHC['subject-id'] + '-' + demoHC['session-id']

# delete duplicates # choose one image per session
fnameformat = '/nfs04/data/OASIS3/image/' + 'sub-' + demoHC['subject-id'] + '/' + \
                demoHC['session-id'] + '/anat/sub-'+ demoHC['subject-id'] + '_' + \
                demoHC['session-id'] + '_'
suffix = []
for i in range(len(fnameformat)):
    suffix.append(demoHC['fname'].iloc[i].split(fnameformat[i])[1].split('.nii.gz')[0])

correct_suffix_list = ['T1w', 'echo-1_run-01_T1w', 'run-01_T1w']
# ['T1w', 'echo-1_run-01_T1w', 'echo-1_run-03_T1w',
#        'echo-2_run-01_T1w', 'echo-2_run-03_T1w', 'run-01_T1w',
#        'run-02_T1w', 'run-03_T1w']

# choose more than one time points
demoHCone = demoHC[np.array(pd.DataFrame(suffix).isin(correct_suffix_list))].reset_index().drop(columns=['index'])
demoHConecount = demoHCone['subject-id'].value_counts() > 1
HClongsubjects = np.array(demoHConecount[demoHConecount].index)
demoHClong = demoHCone[demoHCone['subject-id'].isin(HClongsubjects)].reset_index().drop(columns=['index'])

demoHClong.to_csv('/nfs04/data/OASIS3/demo/demo-healthy-longitudinal.csv')

np.savetxt('/nfs04/data/OASIS3/affine-alignment/imagelist.csv', np.array(demoHClong.fname), '%s')

# train/val/test
demoHClong = pd.read_csv('/nfs04/data/OASIS3/demo/demo-healthy-longitudinal.csv')
HClongsubjects = np.unique(demoHClong['subject-id'])
totalN = len(HClongsubjects)
trainN, valN, testN = int(totalN*0.6), int(totalN*0.2), int(totalN*0.2)
permutedsubjects = HClongsubjects[np.random.permutation(range(totalN))]

trainvaltest = np.zeros(totalN).astype('str')
trainvaltest[:trainN] = 'train'
trainvaltest[trainN:trainN+valN] = 'val'
trainvaltest[trainN+valN:] = 'test'
# assign trainvlatest
demo_trainvaltest = np.zeros(len(demoHClong)).astype('str')
for i in range(len(permutedsubjects)):
    indices = np.where(demoHClong['subject-id'] == permutedsubjects[i])[0]
    demo_trainvaltest[indices] = trainvaltest[i]

demoHClong['trainvaltest'] = demo_trainvaltest
demoHClong.to_csv('/nfs04/data/OASIS3/demo/demo-healthy-longitudinal.csv')

set(demoHClong['subject-id'][demoHClong['trainvaltest'] == 'val'])\
    .intersection(set(demoHClong['subject-id'][demoHClong['trainvaltest'] == 'test']))



# Rigid atlas building
ANTSPATH = '/home/hk672/ANTs/bin/ANTS-build/Examples/'
outputPath = '/nfs04/data/OASIS3/affine-alignment/'
inputPathcsv = '/nfs04/data/OASIS3/affine-alignment/imagelist.csv'

affine_atlas_cmd = f'{ANTSPATH}/antsMultivariateTemplateConstruction2.sh \
              -d 3 \
              -o {outputPath}T_ \
              -i 1 \
              -g 0.25 \
              -j 4 \
              -k 1 \
              -v 10 \
              -c 2 \
              -q 100x100x70x20 \
              -n 0 \
              -r 0 \
              -m MI \
              -l 1 \
              -t Affine \
              {inputPathcsv}'


# get mid save png
sliceidx = 144
outdir = '/nfs04/data/OASIS3/affine-aligned-midslice/images'

for i in range(len(demoHClong)):
    partialname = demoHClong.fname.iloc[i].split('/')[-1].split('.nii.gz')[0]
    fnames = glob.glob('/nfs04/data/OASIS3/affine-alignment/T_template0'+partialname+'*.nii.gz')
    assert len(fnames) == 1
    image = np.asanyarray(nib.load(fnames[0]).dataobj)[:, ::-1, 144].T
    imagerescaled = (((image - image.min()) / (image.max() - image.min())) * 256).astype(np.uint8)
    image2d = Image.fromarray(imagerescaled)
    fname2d = demoHClong['subject-id'].iloc[i] + '_' + demoHClong['session-id'].iloc[i]
    image2d.save(os.path.join(outdir, f'{fname2d}.png'))

demoHClong_slice = pd.read_csv('/nfs04/data/OASIS3/demo/demo-healthy-longitudinal.csv', index_col=[0])
demoHClong_slice.fname = outdir + demoHClong['subject-id'] + '_' + demoHClong['session-id'] + '.png'

# match mriinfo
label = demoHClong_slice['id-session'].str.replace('-ses-', '_MR_')
sex = []
age = []
for l in range(len(label)):
    labelindex = np.where(mriinfo.Label == label.iloc[l])[0]
    sex.append(np.array(mriinfo['M/F'].iloc[labelindex])[0])
    age.append(np.array(mriinfo['Age'].iloc[labelindex])[0])

demoHClong['sex'] = sex
demoHClong['age'] = age
demoHClong_slice['sex'] = sex
demoHClong_slice['age'] = age

demoHClong.to_csv('/nfs04/data/OASIS3/demo/demo-healthy-longitudinal.csv')
demoHClong_slice.to_csv(f'{outdir}/demo-healthy-longitudinal.csv')

# time point
unqID = np.unique(demoHClong['subject-id'])
timepoint = np.zeros(len(demoHClong)).astype('str')
for s in range(len(unqID)):
    subjectidx = np.where(demoHClong['subject-id'] == unqID[s])[0]
    sortedage = np.sort(np.unique(demoHClong['age'].iloc[subjectidx]))
    # save sorted age index
    for sidx in subjectidx:
        timepoint[sidx] = np.where(sortedage == demoHClong['age'].iloc[sidx])[0][0]


demoHClong_slice['timepoint'] = timepoint.astype('int')
demoHClong['timepoint'] = timepoint.astype('int')
demoHClong.to_csv('/nfs04/data/OASIS3/demo/demo-healthy-longitudinal.csv')
demoHClong_slice.to_csv(f'{outdir}/demo-healthy-longitudinal.csv')

demoHClong_slice.fname = 'images/' + demoHClong['subject-id'] + '_' + demoHClong['session-id'] + '.png'
demoHClong_slice.to_csv(f'{outdir}/demo-healthy-longitudinal.csv')



