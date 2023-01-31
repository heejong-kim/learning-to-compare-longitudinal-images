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

def train(network, loader, opt):
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

    os.makedirs("%s/" % (opt.save_name), exist_ok=True)
    if parallel:
        network = nn.DataParallel(network).to(device)

    if opt.epoch > 0 :
        if len(glob.glob("%s/epoch%i*.pth" % (opt.save_name, opt.epoch - 1)))>0:
            lastpointname = glob.glob("%s/epoch%i*.pth" % (opt.save_name, opt.epoch-1))[0]
            network.load_state_dict(torch.load(lastpointname))
            total_iter = int(lastpointname.split(".pth")[0].split('iter')[-1])
        else:
            bestepoch, bestiter = np.loadtxt(os.path.join( opt.save_name, 'best.info'))
            bestpointname = glob.glob("%s/best.pth" % (opt.save_name))[0]
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
    loss_mse = torch.nn.MSELoss()
    if 'scheduler' in opt:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)  # factor 0.1

    steps_per_epoch = opt.num_of_iters  # 781
    writer = SummaryWriter(log_dir="%s" % opt.save_name)

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
                       "%s/epoch%d-iter%d.pth" % (opt.save_name, epoch, total_iter))

        if total_iter > opt.max_iters:
            break

        if earlystoppingcount > 5:
            break

        epoch_total_loss = []
        epoch_step_time = []

        for step, batch in enumerate(loader_train):
            total_iter += 1
            step_start_time = time.time()

            input1, target1 = batch

            optimizer.zero_grad()

            # Feature Network
            pred = network(input1.type(Tensor))
            target = target1[:, None].type(Tensor)

            loss = loss_mse(pred, target)
            loss.backward()
            optimizer.step()

            epoch_total_loss.append(loss.item())


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

                network.eval()
                valloss_total = [];
                for valstep, batch in enumerate(loader_val):
                    input1, target1 = batch
                    pred = network(input1.type(Tensor))
                    target = target1[:, None].type(Tensor)
                    valloss = loss_mse(pred, target)
                    valloss_total.append(valloss.item())
                    valstep += 1

                _log_stats([np.mean(valloss_total)], ['val loss'], total_iter, writer)
                val_loss_info = 'val loss: %.4e' % (np.mean(valloss_total))
                print(' - '.join((epoch_info, time_info, loss_info, val_loss_info)), flush=True)
                network.train()
                curr_val_loss = np.mean(valloss_total)
                if 'scheduler' in opt:
                    scheduler.step(curr_val_loss)
                if prev_val_loss > curr_val_loss:
                    torch.save(network.state_dict(),
                               "%s/best.pth" % (opt.save_name))

                    np.savetxt("%s/best.info" % (opt.save_name), np.array([epoch, total_iter]))
                    prev_val_loss = curr_val_loss
                    earlystoppingcount = 0  # New bottom
                else:
                    earlystoppingcount += 1
                    print(f'Early stopping count: {earlystoppingcount}')

    torch.save(network.state_dict(), "%s/epoch%d-iter%d.pth" % (opt.save_name, epoch, total_iter))
    network.eval()

def test_regression(network, loader, savedmodelname, opt, overwrite=False):
    resultname = f'test-output'

    print('working on ', resultname)
    run = False
    resultfilenmae = os.path.join( opt.save_name, f'{resultname}.csv')
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

        tmp_stack_output = np.empty((0, 1))
        tmp_stack_target = np.empty((0, 1))

        for teststep, batch in enumerate(loader_test):
            sys.stdout.write(
                "\r [Batch %d/%d] "
                % (teststep,
                   len(loader_test),
                   )
            )
            I1 = batch
            input1, target1 = I1

            output = network(input1.type(Tensor))
            tmp_stack_output = np.append(tmp_stack_output, output.cpu().detach().numpy(), axis=0)
            tmp_stack_target = np.append(tmp_stack_target, np.array(target1)[:, None], axis=0)

        result = pd.DataFrame()
        result['output'] = np.array(tmp_stack_output).squeeze()
        result['target'] = np.array(tmp_stack_target).squeeze()
        print('=========================')
        print(opt.save_name, resultname)
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
parser.add_argument('--num_of_iters', default=200, type=int, help="number of iteration for validation")
parser.add_argument('--epoch', default=0, type=int, help="Starting epoch")
parser.add_argument('--num_workers', default=12, type=int)

parser.add_argument('--image_size', default="68,68", type=str, help="x,y")
parser.add_argument('--image_channel', default=1, type=int)
parser.add_argument('--image_dir', default='./datasets/starmen-augmentation', type=str)
parser.add_argument('--targetname', default='tstar', type=str)
parser.add_argument('--dataname', default='starmen', type=str)


opt = parser.parse_args()
set_manual_seed(opt.seed)
suffix = f'seed{torch.initial_seed()}'

image_size = [int(item) for item in opt.image_size.split(',')]
opt.image_size = image_size

dict_dataloader = {'starmen': STARMENregression, 'tumor': TUMORregression,
                   'embryo': EMBRYOregression, 'oasis': OASISregression}

if __name__ == "__main__":

    # train baseline regression assuming cross-crosssectional data
    opt.save_name = f'result-model/{opt.dataname}/lr{opt.lr}-b1{opt.b1}-b2{opt.b2}{suffix}/' \
                        f'PaIRNet-self-supervised'

    network = Resnet18Regression(channels=opt.image_channel)
    train(network, dict_dataloader[opt.dataname], opt)

    #
    # # test the trained model with the best weight
    # savedmodelname = os.path.join(opt.save_name, 'best.pth')
    # result = test_regression(network, dict_dataloader[opt.dataname], savedmodelname, opt, overwrite=False)
    #

