import os
from torchvision import datasets
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import datetime
import time
import sys
from tensorboardX import SummaryWriter
import itertools as it
import argparse
import torchvision.models as models
import random
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import glob
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import roc_auc_score as auc
import matplotlib.pyplot as plt
from scipy.io import loadmat
# import mat73
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix

from loader import *
from model import *
from utils import *
from utils import _log_stats


def correlation_test(result_delta_time_corr, figure=True, figurename=''):
    plt.rcParams.update({'font.size': 20})

    x = 'gt-target'
    y = 'feature'
    result_pos = result_delta_time_corr[result_delta_time_corr.target >= 0].reset_index()
    xx = np.array(result_pos[x])
    yy = np.array(result_pos[y])
    r, p = pearsonr(xx, yy)

    if figure:
        plt.close();
        plt.clf()
        fig, arr = plt.subplots(1, 1)
        arr.scatter(xx, yy, marker='+', color=['k'])
        plt.title(f'Pearson:{r:.2f}')
        fig.savefig(figurename)

    return r, p

def test_PaIRNet(network, loader, savedmodelname, opt, gt_target, subjidname, overwrite=False):
    print('working on ', savedmodelname)
    run = False
    resultfilename = os.path.join(opt.save_name.split('.pth')[0] + f'-test-prediction.csv')
    if not os.path.exists(resultfilename) or overwrite:
        run = True
    else:
        result = pd.read_csv(resultfilename)
        loader_test = torch.utils.data.DataLoader(
            loader(root=opt.image_dir, trainvaltest='test', transform=False, opt=opt),
            batch_size=64, shuffle=False, num_workers=opt.num_workers)

    if run:
        cuda = True
        parallel = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        activation_sigmoid = nn.Sigmoid().cuda()

        if parallel:
            network = nn.DataParallel(network).to(device)
            network.load_state_dict(torch.load(savedmodelname))
        else:
            network.load_state_dict(torch.load(savedmodelname))
            if cuda:
                network = network.cuda()

        network.eval()

        loader_test = torch.utils.data.DataLoader(
            loader(root=opt.image_dir, trainvaltest='test', transform=False, opt=opt),
            batch_size=64, shuffle=False, num_workers=opt.num_workers)

        targetdiffvalue = np.empty((0, 1))
        featurevalue = np.empty((0, 1))

        for teststep, batch in enumerate(loader_test):
            sys.stdout.write(
                "\r [Batch %d/%d] "
                % (teststep,
                   len(loader_test),
                   )
            )
            I1, I2 = batch
            input1, target1 = I1
            input2, target2 = I2

            featurevalue_tmp = network(input1.type(Tensor), input2.type(Tensor))
            targetdiff = ((target1 - target2)[:, None]).type(Tensor)
            targetdiffvalue = np.append(targetdiffvalue, np.array(targetdiff.cpu().detach()), axis=0)
            featurevalue = np.append(featurevalue,
                                              featurevalue_tmp.cpu().detach(), axis=0)

        result = pd.DataFrame()
        result['target'] = targetdiffvalue.tolist()
        result['feature'] = featurevalue.tolist()
        result['target'] = np.array(targetdiffvalue)
        result['feature'] = np.array(featurevalue)
        result['pairindex1'] = loader_test.dataset.index_combination[:, 0]
        result['pairindex2'] = loader_test.dataset.index_combination[:, 1]

        gt_target_diff = np.array(loader_test.dataset.demo[gt_target].iloc[loader_test.dataset.index_combination[:, 0]]) - \
                         np.array(loader_test.dataset.demo[gt_target].iloc[loader_test.dataset.index_combination[:, 1]])

        result['gt-target'] = gt_target_diff
        result['gt-target-base'] = np.array(loader_test.dataset.demo[gt_target].iloc[loader_test.dataset.index_combination[:, 0]])
        if 'fname' in list(loader_test.dataset.demo.columns):
            result['fname-pairindex1'] = np.array(loader_test.dataset.demo['fname'].iloc[loader_test.dataset.index_combination[:, 0]])
            result['fname-pairindex2'] = np.array(loader_test.dataset.demo['fname'].iloc[loader_test.dataset.index_combination[:, 1]])
        elif 'path' in list(loader_test.dataset.demo.columns):
            result['fname-pairindex1'] = np.array(loader_test.dataset.demo['path'].iloc[loader_test.dataset.index_combination[:, 0]])
            result['fname-pairindex2'] = np.array(loader_test.dataset.demo['path'].iloc[loader_test.dataset.index_combination[:, 1]])

        result['subjidname'] = np.array(loader_test.dataset.demo[subjidname].iloc[loader_test.dataset.index_combination[:, 1]])
        result.to_csv(resultfilename)

    return result, loader_test.dataset.demo

def test_crosssectional_regression(network, loader, savedmodelname, opt, overwrite=False):
    print(f'Working on {savedmodelname}')
    run = False
    resultfilename = os.path.join(opt.save_name.split('.pth')[0] + f'-test-prediction.csv')
    if not os.path.exists(resultfilename) or overwrite:
        run = True
    else:
        result = pd.read_csv(resultfilename)
        loader_test = torch.utils.data.DataLoader(
            loader(root=opt.image_dir, trainvaltest='test', transform=False, opt=opt),
            batch_size=64, shuffle=False, num_workers=opt.num_workers)

    if run:
        cuda = True
        parallel = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        activation_sigmoid = nn.Sigmoid().cuda()

        if parallel:
            network = nn.DataParallel(network).to(device)
            network.load_state_dict(torch.load(savedmodelname))
        else:
            network.load_state_dict(torch.load(savedmodelname))
            if cuda:
                network = network.cuda()

        network.eval()

        loader_test = torch.utils.data.DataLoader(
            loader(root=opt.image_dir, trainvaltest='test', transform=False, opt=opt),
            batch_size=64, shuffle=False, num_workers=opt.num_workers)

        targetvalue = np.empty((0, 1))
        featurevalue = np.empty((0, 1))

        for teststep, batch in enumerate(loader_test):
            sys.stdout.write(
                "\r [Batch %d/%d] "
                % (teststep,
                   len(loader_test),
                   )
            )
            input1, target1 = batch[0], batch[1]

            featurevalue_tmp = network(input1.type(Tensor))
            targetvalue = np.append(targetvalue, np.array(target1.cpu().detach())[:, None], axis=0)
            featurevalue = np.append(featurevalue,
                                              featurevalue_tmp.cpu().detach(), axis=0)

        result = pd.DataFrame()
        result['target'] = targetvalue.tolist()
        result['feature'] = featurevalue.tolist()
        result['target'] = np.array(targetvalue)
        result['feature'] = np.array(featurevalue)
        result.to_csv(resultfilename)

    return result, loader_test.dataset.demo


dict_dataloader = {'starmen': STARMEN, 'tumor': TUMOR,
                   'embryo': EMBRYO, 'oasis': OASIS}
dict_subjectname = {'embryo':'embryoname', 'tumor': 'Subject ID', 'starmen': 'id','oasis': 'subject-id'}


parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--image_size', default="68,68", type=str, help="x,y", required=True)
parser.add_argument('--image_channel', default=1, type=int)
parser.add_argument('--image_dir', default='./datasets/starmen-augmentation', type=str)
parser.add_argument('--dataname', type=str, required=True)
parser.add_argument('--save_name', type=str, required=True, help="path to saved model (.pth)")
parser.add_argument('--targetname', type=str, required=True, help="groundtruth target name (x axis)")

opt = parser.parse_args()
image_size = [int(item) for item in opt.image_size.split(',')]
opt.image_size = image_size


if __name__ == "__main__":

    network = Resnet18Diff(channels=opt.image_channel)
    savedmodelname = os.path.join(opt.save_name)
    result_delta_time, demo = test_PaIRNet(network, dict_dataloader[opt.dataname], savedmodelname, opt, opt.targetname,
                                                dict_subjectname[opt.dataname], overwrite=False)
    if 'starmen' in opt.dataname:
        t_star = np.array(demo["alpha"] * (demo["t"] - demo["tau"]))
        t_star[np.array(result_delta_time.pairindex1).astype('int')] - t_star[np.array(result_delta_time.pairindex2).astype('int')]
        result_delta_time['gt-target'] = t_star[np.array(result_delta_time.pairindex1).astype('int')] - t_star[np.array(result_delta_time.pairindex2).astype('int')]
        result_delta_time['gt-target-base'] = t_star[np.array(result_delta_time.pairindex1).astype('int')]

    os.makedirs('./result-correlation')
    r, p = correlation_test(result_delta_time, figure=True, figurename=f'./result-correlation/{opt.dataname}-{opt.targetname}.png')


    ## --- for regression
    #
    # network = Resnet18Regression(channels=opt.image_channel)
    # savedmodelname = os.path.join(opt.save_name)
    # result_reg_pred, demo = test_crosssectional_regression(network, dict_dataloader[opt.dataname], savedmodelname, opt, overwrite=False)
    #
    # IDunq = np.unique(demo[dict_subjectname[opt.dataname]])
    # index_combination = np.empty((0, 2))
    # for sid in IDunq:
    #     indices = np.where(demo[dict_subjectname[opt.dataname]] == sid)[0]
    #     ### all possible pairs
    #     tmp_combination = np.array(
    #         np.meshgrid(np.array(range(len(indices))), np.array(range(len(indices))))).T.reshape(-1, 2)
    #     index_combination = np.append(index_combination, indices[tmp_combination], 0)
    #
    # index_combination = index_combination.astype('int')
    # result_delta_time = pd.DataFrame()
    # result_delta_time['gt-target'] = np.array(result_reg_pred['target'])[index_combination[:, 0]] - \
    #                                         np.array(result_reg_pred['target'])[index_combination[:, 1]]
    # result_delta_time['feature'] = np.array(result_reg_pred['feature'])[index_combination[:, 0]] - \
    #                                         np.array(result_reg_pred['feature'])[index_combination[:, 1]]
    # result_delta_time['target'] = result_delta_time['gt-target']
    #
    # plt.rcParams.update({'font.size': 20})
    #
    # r, p = correlation_test(result_delta_time, figure=True, figurename=f'./figure-correlation-{task}-{savename}.png')
    #

