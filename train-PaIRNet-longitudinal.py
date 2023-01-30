from torch.utils.data import Dataset
import sys
from tensorboardX import SummaryWriter
import argparse
import pandas as pd
import glob
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import roc_auc_score as auc
from loader import *
from model import *
from utils import *
import torch
import numpy as np
import random
import os
from utils import _log_stats
import time
import datetime

'''
trainer
'''

def train(network, loader, opt, selfsupervised=True):

    print(opt)
    cuda = True
    parallel = True
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    if 'initialization' in opt:
        if 'features' in dir(network):
            init_weights(network.features, type=opt.initialization)
        if 'classifier' in dir(network):
            init_weights(network.classifier, type=opt.initialization)
        if 'fc' in dir(network):
            init_weights(network.fc, type=opt.initialization)

        print(f'Initialized: {opt.initialization}')

    os.makedirs("saved_models/%s/" % (opt.save_name), exist_ok=True)
    if parallel:
        network = nn.DataParallel(network).to(device)

    if opt.epoch > 0 :
        if len(glob.glob("saved_models/%s/epoch%i*.pth" % (opt.save_name, opt.epoch - 1)))>0:
            lastpointname = glob.glob("saved_models/%s/epoch%i*.pth" % (opt.save_name, opt.epoch-1))[0]
            network.load_state_dict(torch.load(lastpointname))
            total_iter = int(lastpointname.split(".pth")[0].split('iter')[-1])
        else:
            bestepoch, bestiter = np.loadtxt(os.path.join('saved_models/' + opt.save_name, 'best.info'))
            bestpointname = glob.glob("saved_models/%s/best.pth" % (opt.save_name))[0]
            network.load_state_dict(torch.load(bestpointname))
            opt.epoch = int(bestepoch)
            total_iter = bestiter
        assert total_iter > 0, "total iteration should start higher when opt.epoch > 0 "
    else:
        total_iter = 0


    print("=========================================")
    print("Num of param:", count_parameters(network))
    print("=========================================")

    optimizer = torch.optim.Adam(network.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    loss_bcelogit = torch.nn.BCEWithLogitsLoss()
    if 'scheduler' in opt:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)  # factor 0.1

    steps_per_epoch = opt.max_iters  # 781
    writer = SummaryWriter(log_dir="saved_models/%s" % opt.save_name)

    prev_time = time.time()
    prev_val_loss = 400
    earlystoppingcount = 0

    loader_train = torch.utils.data.DataLoader(  #
        loader(root=opt.image_dir, trainvaltest='train', transform=True, opt=opt),
        batch_size=64, shuffle=True, num_workers=opt.num_workers, drop_last=True)

    loader_val = torch.utils.data.DataLoader(
        loader(root=opt.image_dir, trainvaltest='val', transform=False, opt=opt),
        batch_size=64, shuffle=True, num_workers=opt.num_workers, drop_last=True)

    for epoch in range(opt.epoch, opt.max_epoch):

        if epoch == int(opt.max_epoch / 4):
            torch.save(network.state_dict(),
                       "saved_models/%s/epoch%d-iter%d.pth" % (opt.save_name, epoch, total_iter))

        if total_iter > opt.max_iters:
            break

        if earlystoppingcount > 5:
            break

        epoch_total_loss = []
        epoch_step_time = []
        epoch_total_acc = []

        for step, batch in enumerate(loader_train):
            total_iter += 1
            step_start_time = time.time()

            # Model inputs
            I1, I2 = batch
            input1, target1 = I1
            input2, target2 = I2

            optimizer.zero_grad()

            # Feature Network
            featureDiff = network(input1.type(Tensor), input2.type(Tensor))
            targetdiff = ((target1 - target2)[:, None]).type(Tensor)
            if selfsupervised:
                targetdiff[targetdiff > 0] = 1
                targetdiff[targetdiff == 0] = 0.5
                targetdiff[targetdiff < 0] = 0

            loss = loss_bcelogit(featureDiff, targetdiff)
            loss.backward()
            optimizer.step()

            epoch_total_loss.append(loss.item())
            epoch_total_acc.append(np.sum(np.array(featureDiff.cpu().detach() > 0.5) == \
                                          np.array(targetdiff.cpu().detach() == 1)) / len(targetdiff))

            # --------------
            #  Log Progress
            # --------------
            # Determine approximate time left
            batches_done = epoch * len(loader_train) + step
            batches_left = opt.max_epoch * len(loader_train) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [ loss: %f ] ETA: %s"
                % (
                    epoch,
                    opt.max_epoch,
                    step,
                    len(loader_train),
                    loss.item(),
                    time_left,
                )
            )
            epoch_step_time.append(time.time() - step_start_time)

            # validation
            if (total_iter % steps_per_epoch == 0):
                epoch_info = '\nValidating... Step %d/%d / Epoch %d/%d' % (
                    step, len(loader_train), epoch, opt.max_epoch)
                time_info = '%.4f sec/step' % np.mean(epoch_step_time)
                loss_info = 'train loss: %.4e ' % (np.mean(epoch_total_loss))

                _log_stats([np.mean(epoch_total_loss)], ['train_loss'], total_iter, writer)
                _log_stats([np.mean(epoch_total_acc)], ['train_acc'], total_iter, writer)

                network.eval()
                valloss_total = [];
                epoch_val_acc = []
                for valstep, batch in enumerate(loader_val):
                    I1, I2 = batch
                    input1, target1 = I1
                    input2, target2 = I2
                    featureDiff = network(input1.type(Tensor), input2.type(Tensor))

                    targetdiff = ((target1 - target2)[:, None]).type(Tensor)
                    targetdiff[targetdiff > 0] = 1
                    targetdiff[targetdiff == 0] = 0.5
                    targetdiff[targetdiff < 0] = 0

                    valloss = loss_bcelogit(featureDiff, targetdiff)
                    epoch_val_acc.append(np.sum(np.array(featureDiff.cpu().detach() > 0.5) == \
                                                np.array(targetdiff.cpu().detach() == 1)) / len(targetdiff))
                    valloss_total.append(valloss.item())
                    valstep += 1

                _log_stats([np.mean(valloss_total)], ['val loss'], total_iter, writer)
                _log_stats([np.mean(epoch_val_acc)], ['val_acc'], total_iter, writer)
                val_loss_info = 'val loss: %.4e' % (np.mean(valloss_total))
                print(' - '.join((epoch_info, time_info, loss_info, val_loss_info)), flush=True)
                network.train()
                curr_val_loss = np.mean(valloss_total)
                if 'scheduler' in opt:
                    scheduler.step(curr_val_loss)
                if prev_val_loss > curr_val_loss:
                    torch.save(network.state_dict(),
                               "saved_models/%s/best.pth" % (opt.save_name))

                    np.savetxt("saved_models/%s/best.info" % (opt.save_name), np.array([epoch, total_iter]))
                    prev_val_loss = curr_val_loss
                    earlystoppingcount = 0  # New bottom
                else:
                    earlystoppingcount += 1
                    print(f'Early stopping count: {earlystoppingcount}')

    torch.save(network.state_dict(), "saved_models/%s/epoch%d-iter%d.pth" % (opt.save_name, epoch, total_iter))
    network.eval()

def test(network, loader, savedmodelname, opt, overwrite=False):
    resultname = f'test-all-repeat{opt.num_repeat}-epoch{opt.max_epoch}'
    print('working on ', resultname)
    run = False
    resultfilenmae = os.path.join('saved_models/' + opt.save_name, f'{resultname}.csv')
    if os.path.exists(resultfilenmae):
        print(f'{resultname} EXISTS')
        print(f'....result loaded from the exisintg file')
        result = pd.read_csv(resultfilenmae)

    if not os.path.exists(resultfilenmae) or overwrite:
        run = True

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
            batch_size=64, shuffle=True, num_workers=opt.num_workers)

        stack_acc = []
        stack_auc = []
        stack_acc01 = []
        stack_auc01 = []
        stack_FP = []
        stack_TP = []
        stack_FN = []
        stack_TN = []
        stack_Tsame_01 = []
        stack_Tsame_001 = []
        stack_totalsame = []
        stack_totalnum = []
        stack_l1 = []

        tmp_stack_feature_diff = np.empty((0, 1))
        # tmp_stack_feature_diff_rev = np.empty((0, 1))
        tmp_stack_target_diff = np.empty((0, 1))
        tmp_stack_target1 = np.empty((0, 1))
        tmp_stack_target2 = np.empty((0, 1))

        for teststep, batch in enumerate(loader_test):
            sys.stdout.write(
                "\r [Batch %d/%d] "  # [ target diff: %d ]
                % (teststep,
                   len(loader_test),
                   )
            )
            I1, I2 = batch
            input1, target1 = I1
            input2, target2 = I2
            featurediff = network(input1.type(Tensor), input2.type(Tensor))
            # featurediffrev = network(input2.type(Tensor), input1.type(Tensor))
            targetdiff = ((target1 - target2)[:, None]).type(Tensor)
            targetdiff[targetdiff > 0] = 1
            targetdiff[targetdiff == 0] = 0.5
            targetdiff[targetdiff < 0] = 0
            tmp_stack_feature_diff = np.append(tmp_stack_feature_diff,
                                               np.array((activation_sigmoid(featurediff)).cpu().detach()),
                                               axis=0)
            tmp_stack_target_diff = np.append(tmp_stack_target_diff,
                                              targetdiff.cpu().detach(), axis=0)
            tmp_stack_target1 = np.append(tmp_stack_target1, np.array(target1)[:, None], axis=0)
            tmp_stack_target2 = np.append(tmp_stack_target2, np.array(target2)[:, None], axis=0)

        # make one hot encoding
        onehot_target = np.hstack((tmp_stack_target_diff, 1 - tmp_stack_target_diff))
        onehot_feature = np.hstack((tmp_stack_feature_diff, 1 - tmp_stack_feature_diff))
        accval = acc(tmp_stack_target_diff > 0.5, tmp_stack_feature_diff > 0.5)
        aucval = auc(tmp_stack_target_diff > 0.5, tmp_stack_feature_diff)
        accval01 = acc(tmp_stack_target_diff[tmp_stack_target_diff != 0.5],
                       tmp_stack_feature_diff[tmp_stack_target_diff != 0.5] > 0.5)
        aucval01 = auc(tmp_stack_target_diff[tmp_stack_target_diff != 0.5],
                       tmp_stack_feature_diff[tmp_stack_target_diff != 0.5])
        FP = np.logical_and(tmp_stack_target_diff == 0, (tmp_stack_feature_diff > 0.5) == 1)
        FN = np.logical_and(tmp_stack_target_diff == 1, (tmp_stack_feature_diff > 0.5) == 0)
        totalsame = np.sum(tmp_stack_target_diff == 0.5)

        TP = np.logical_and(tmp_stack_target_diff == 1, (tmp_stack_feature_diff > 0.5) == 1)
        TN = np.logical_and(tmp_stack_target_diff == 0, (tmp_stack_feature_diff > 0.5) == 0)
        Tsame_thr01 = np.logical_and(tmp_stack_target_diff == 0.5,
                                     np.logical_and(tmp_stack_feature_diff < 0.51,
                                                    tmp_stack_feature_diff > 0.49))
        Tsame_thr001 = np.logical_and(tmp_stack_target_diff == 0.5,
                                      np.logical_and(tmp_stack_feature_diff < 0.501,
                                                     tmp_stack_feature_diff > 0.499))

        stack_acc.append(np.sum(accval))
        stack_auc.append(np.sum(aucval))
        stack_acc01.append(np.sum(accval01))
        stack_auc01.append(np.sum(aucval01))
        stack_FP.append(np.sum(FP))
        stack_TP.append(np.sum(TP))
        stack_FN.append(np.sum(FN))
        stack_TN.append(np.sum(TN))
        stack_Tsame_01.append(np.sum(Tsame_thr01))
        stack_Tsame_001.append(np.sum(Tsame_thr001))
        stack_totalsame.append(totalsame)
        stack_totalnum.append(len(FP))
        stack_l1.append(np.mean(np.abs(onehot_target - onehot_feature)))

        result = pd.DataFrame()
        result['auc'] = np.array(stack_auc)
        result['acc'] = np.array(stack_acc)
        result['auc01'] = np.array(stack_auc01)
        result['acc01'] = np.array(stack_acc01)
        result['FP'] = np.array(stack_FP)
        result['TP'] = np.array(stack_TP)
        result['FN'] = np.array(stack_FN)
        result['TN'] = np.array(stack_TN)
        result['Tsame_thr01'] = np.array(stack_Tsame_01)
        result['Tsame_thr001'] = np.array(stack_Tsame_001)
        result['num_same_label'] = np.array(stack_totalsame)
        result['num_total'] = np.array(stack_totalnum)
        result['l1'] = np.array(stack_l1)
        # result['l1avg'] = np.array(stack_l1avg)
        print('=========================')
        print(opt.save_name, resultname)
        meanresult = round(result.mean(axis=0), 3)
        print(f'\tL1: {meanresult.l1} /   \n'  # L1avg: {meanresult.l1avg}
              f'\tAUC: {meanresult.auc} / ACC: {meanresult.acc} / '
              f'AUC01: {meanresult.auc01} / ACC01: {meanresult.acc01} '
              f' Total same: {meanresult.num_same_label}')
        result.to_csv(resultfilenmae)

    return result

parser = argparse.ArgumentParser()


parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--b1', default=0.9, type=float)
parser.add_argument('--b2', default=0.999, type=float)
parser.add_argument('--initialization', default='kaiming', type=str)
parser.add_argument('--seed', default=0, type=int)

parser.add_argument('--max_epoch', default=200, type=int, help="Max epoch")
parser.add_argument('--max_iters', default=10000000000, type=int, help="Max iteration")
parser.add_argument('--epoch', default=0, type=int, help="Starting epoch")
parser.add_argument('--num_workers', default=12, type=int)

parser.add_argument('--image_size', default="68,68", type=str, help="x,y", required=True)
parser.add_argument('--image_channel', default=1, type=int)
parser.add_argument('--image_dir', default='./datasets/starmen-augmentation', type=str)
parser.add_argument('--targetname', default='timepoint', type=str)
parser.add_argument('--dataname', default='starmen', type=str)
parser.add_argument('--selfsupervised', action=argparse.BooleanOptionalAction)


opt = parser.parse_args()
set_manual_seed(opt.seed)
suffix = f'seed{torch.initial_seed()}'

image_size = [int(item) for item in opt.image_size.split(',')]
opt.image_size = image_size

dict_dataloader = {'starmen': STARMEN, 'tumor': TUMOR,
                   'embryo': EMBRYO, 'oasis': OASIS}

if __name__ == "__main__":

    # train PaIRNet
    if opt.selfsupervised:
        opt.save_name = f'result/{opt.dataname}/lr{opt.lr}-b1{opt.b1}-b2{opt.b2}{suffix}/' \
                        f'PaIRNet-self-supervised'
    else:
        opt.save_name = f'result/{opt.dataname}/lr{opt.lr}-b1{opt.b1}-b2{opt.b2}{suffix}/' \
                        f'PaIRNet-supervised'

    network = Resnet18Diff(channels=opt.image_channel)
    train(network, dict_dataloader[opt.dataname], opt, selfsupervised=opt.selfsupervised)

    #
    # # test the trained model with the best weight
    # savedmodelname = os.path.join('saved_models/' + opt.save_name, 'best.pth')
    # result = test(network, dict_dataloader[opt.dataname], savedmodelname, opt, overwrite=False)
    #




# python ./train-PaIRNet-longitudinal.py --max_epoch=1 --num_workers=1 --image_size="68,68" --image_channel=1 --image_dir='/scratch/datasets/hk672/starmen-augmentation' --dataname='starmen' --selfsupervised
