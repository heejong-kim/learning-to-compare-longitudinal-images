import matplotlib.pyplot as plt
from loader import *
from model import *
from utils import *
import argparse
import torch.nn as nn
import os
import numpy as np
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
import cv2

cuda = True
parallel = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def get_featurewise_gradcam(input, model, featureidx):
    model.zero_grad()
    input_tensor = torch.autograd.Variable(input, requires_grad=True)
    feature = model(input_tensor[None, :])
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    target_categories = np.array([featureidx])
    targets = [ClassifierOutputTarget(
        category) for category in target_categories]

    loss = sum([target(output)
                for target, output in zip(targets, feature)])
    loss.backward(retain_graph=True)
    grad = model.module.gradients[0].cpu().data.numpy()
    activations = model.module.activations[0].cpu().data.numpy()
    model.module.release()
    return grad, activations

def get_cam_image(weighted_activations, input):
    targetsize = [input.shape[-2], input.shape[-1]]
    cam = weighted_activations.sum(axis=1)
    cam_min = np.min(cam)
    cam = cam - cam_min
    cam_max = np.max(cam)
    cam = cam / (1e-7 + cam_max)
    scaled = cv2.resize(cam[0], targetsize[::-1])
    inputvis = input.numpy().transpose(1, 2, 0)
    inputvis = np.minimum(np.maximum(inputvis, 0), 1)
    cam_image = show_cam_on_image(inputvis,
                                  scaled.squeeze(), use_rgb=True)
    return cam_image, inputvis

def load_pair_model(visnetwork, savedmodelname):

    # load weight from original network
    tmpweight = torch.load(savedmodelname)
    tmpkeys = list(tmpweight.keys())
    for k in tmpkeys:
        if 'features' in k:
            f1 = k.replace('features', 'features1')
            f2 = k.replace('features', 'features2')
            tmpweight[f1] = tmpweight[k]
            tmpweight[f2] = tmpweight[k]
            del tmpweight[k]

    if parallel:
        visnetwork = nn.DataParallel(visnetwork).to(device)
        visnetwork.load_state_dict(tmpweight)
    else:
        visnetwork.load_state_dict(tmpweight)
        if cuda:
            visnetwork = visnetwork.cuda(tmpweight)

    visnetwork.eval()
    model = visnetwork
    return model

def load_single_model(visnetwork, savedmodelname):

    # load weight from original network
    tmpweight = torch.load(savedmodelname)
    tmpkeys = list(tmpweight.keys())
    for k in tmpkeys:
        if 'features' in k:
            f1 = k.replace('features', 'features1')
            tmpweight[f1] = tmpweight[k]
            del tmpweight[k]

    if parallel:
        visnetwork = nn.DataParallel(visnetwork).to(device)
        visnetwork.load_state_dict(tmpweight)
    else:
        visnetwork.load_state_dict(tmpweight)
        if cuda:
            visnetwork = visnetwork.cuda(tmpweight)

    visnetwork.eval()
    model = visnetwork
    return model

def _visualize_all_pair(model, vissavedir, loader_test, subjidname):

    demo = loader_test.dataset.demo
    unqid = np.unique(demo[subjidname])
    pairindices = loader_test.dataset.index_combination.astype('int')
    count = 0
    subjectN = len(unqid)

    for uid in unqid:
        count += 1
        if count % int(subjectN/10) == 0 :
            print(f'{count} out of {subjectN}')

        vis_index = np.array(demo.index[demo[subjidname] == uid]).astype('int')

        if len(vis_index) >= 2:
            plt.close()
            plt.clf()
            fig, arr = plt.subplots(len(vis_index), len(vis_index), figsize=(30, 30))

            for v in range(len(vis_index)):
                for w in range(len(vis_index)):
                        if w >= v:

                            pairidx =np.where(np.sum(pairindices == [vis_index[v], vis_index[w]], 1) == 2)[0][0]
                            I1, I2 = loader_test.dataset.__getitem__(pairidx)
                            input1, target1 = I1
                            input2, target2 = I2

                            input1 = torch.tensor(input1)
                            input2 = torch.tensor(input2)

                            f1 = model(input1[None, :])
                            f2 = model(input2[None, :])
                            feature_weight = (f1 - f2).cpu().detach().numpy()
                            linear_weight = model.module.classifier[0].weight.cpu().detach().numpy()
                            total_weight = np.abs(feature_weight*linear_weight).squeeze()
                            total_weight_sign = (feature_weight*linear_weight).squeeze()

                            _, activation1 = get_featurewise_gradcam(input1, model, 0)
                            _, activation2 = get_featurewise_gradcam(input2, model, 0)
                            # weight the activation
                            total_weight_reshape = np.repeat(np.repeat(total_weight[None, :, None, None], activation1.shape[-2], 2), activation1.shape[-1], 3)
                            weighted_activation1 = activation1 * total_weight_reshape
                            weighted_activation2 = activation2 * total_weight_reshape

                            cam1, inputvis1 = get_cam_image(weighted_activation1, input1)
                            cam2, inputvis2 = get_cam_image(weighted_activation2, input2)

                            if inputvis1.shape[2] == 1:
                                inputvis1 = np.repeat(inputvis1, 3, 2)
                                inputvis2 = np.repeat(inputvis2, 3, 2)

                            model.module.release()
                            output = np.sum(feature_weight * linear_weight)
                            arr[v][w].set_title(f'outputs:{output:.2f} \n target:{target1 - target2:.2f}')
                            arr[v][w].imshow(np.concatenate((np.concatenate((cam1/255, cam2/255), 1),
                                                            np.concatenate((inputvis1, inputvis2), 1)), 0))

                        arr[v][w].axis('off')

            fig.savefig(f'{vissavedir}/{uid}.svg')

def _visualize_t0_pair(model, vissavedir, loader_test, subjidname):


    demo = loader_test.dataset.demo
    unqid = np.unique(demo[subjidname])
    pairindices = loader_test.dataset.index_combination.astype('int')
    count = 0
    subjectN = len(unqid)

    for uid in unqid:
        count += 1
        if count % int(subjectN/10) == 0 :
            print(f'{count} out of {subjectN}')

        vis_index = np.array(demo.index[demo[subjidname] == uid]).astype('int')
        if len(vis_index) >= 2:
            plt.close()
            plt.clf()
            fig, arr = plt.subplots(4, len(vis_index), figsize=(30, 12))

            v = 0
            for w in range(len(vis_index)):

                pairidx =np.where(np.sum(pairindices == [vis_index[v], vis_index[w]], 1) == 2)[0][0]
                I1, I2 = loader_test.dataset.__getitem__(pairidx)
                input1, target1 = I1
                input2, target2 = I2

                input1 = torch.tensor(input1)
                input2 = torch.tensor(input2)

                f1 = model(input1[None, :])
                f2 = model(input2[None, :])
                feature_weight = (f1 - f2).cpu().detach().numpy()
                linear_weight = model.module.classifier[0].weight.cpu().detach().numpy()
                total_weight = np.abs(feature_weight*linear_weight).squeeze()

                _, activation1 = get_featurewise_gradcam(input1, model, 0)
                _, activation2 = get_featurewise_gradcam(input2, model, 0)
                # weight the activation
                total_weight_reshape = np.repeat(np.repeat(total_weight[None, :, None, None], activation1.shape[-2], 2), activation1.shape[-1], 3)
                weighted_activation1 = activation1 * total_weight_reshape
                weighted_activation2 = activation2 * total_weight_reshape

                cam1, inputvis1 = get_cam_image(weighted_activation1, input1)
                cam2, inputvis2 = get_cam_image(weighted_activation2, input2)

                if inputvis1.shape[2] == 1:
                    inputvis1 = np.repeat(inputvis1, 3, 2)
                    inputvis2 = np.repeat(inputvis2, 3, 2)


                model.module.release()
                output = np.sum(feature_weight * linear_weight)
                arr[0][w].set_title(f'outputs:{output:.2f} \n target:{target1-target2:.2f}')

                arr[0][w].imshow(cam1)
                arr[1][w].imshow(inputvis1)
                arr[2][w].imshow(cam2)
                arr[3][w].imshow(inputvis2)
                arr[0][w].axis('off')
                arr[1][w].axis('off')
                arr[2][w].axis('off')
                arr[3][w].axis('off')

            fig.savefig(f'{vissavedir}/{uid}.svg')

def _visualize_regression(model, vissavedir, loader_test, subjidname):

    demo = loader_test.dataset.demo
    unqid = np.unique(demo[subjidname])
    count = 0
    subjectN = len(unqid)
    # if subjectN > 150:
    #     unqid = unqid[:150]

    for uid in unqid:
        count += 1
        if count % int(subjectN/10) == 0 :
            print(f'{count} out of {subjectN}')

        vis_index = np.array(demo.index[demo[subjidname] == uid]).astype('int')
        if len(vis_index) >= 2:
            plt.close()
            plt.clf()
            fig, arr = plt.subplots(2, len(vis_index), figsize=(30, 6))

            for w in range(len(vis_index)):
                model.zero_grad()
                I1 = loader_test.dataset.__getitem__(vis_index[w])
                input1, target1 = I1
                target = target1
                input1 = torch.tensor(input1)

                feature_weight = model(input1[None, :]).cpu().detach().numpy()
                linear_weight = model.module.classifier[0].weight.cpu().detach().numpy()
                total_weight = np.abs(feature_weight * linear_weight).squeeze()

                _, activation1 = get_featurewise_gradcam(input1, model, 0)

                # weight the activation
                total_weight_reshape = np.repeat(np.repeat(total_weight[None, :, None, None], activation1.shape[-2], 2),
                                                 activation1.shape[-1], 3)
                weighted_activation1 = activation1 * total_weight_reshape

                cam1, inputvis1 = get_cam_image(weighted_activation1, input1)

                if inputvis1.shape[2] == 1:
                    inputvis1 = np.repeat(inputvis1, 3, 2)

                model.module.release()
                output = np.sum(feature_weight * linear_weight)
                arr[0][w].set_title(f'outputs:{output:.2f} \n target:{target:.2f}')
                arr[0][w].imshow(cam1)
                arr[1][w].imshow(inputvis1)
                arr[0][w].axis('off')
                arr[1][w].axis('off')

            fig.savefig(f'{vissavedir}/{uid}.svg') # ./ layercam / fig-shapes-size

def visualize_PaIRNet(loader, opt, n_channels, subjidname, savename, overwrite = False, all_or_t0 = 't0'):
    loader_test = torch.utils.data.DataLoader(
        loader(root=opt.image_dir, trainvaltest='test', transform=False, opt=opt),
        batch_size=1, shuffle=False, num_workers=opt.num_workers)

    savedmodelname = os.path.join(opt.save_name)
    tmpweight = torch.load(savedmodelname)
    visnetwork = Resnet18DiffForPairVisFeature(channels=n_channels)

    if parallel:
        visnetwork = nn.DataParallel(visnetwork).to(device)
        visnetwork.load_state_dict(tmpweight)
    else:
        visnetwork.load_state_dict(tmpweight)
        if cuda:
            visnetwork = visnetwork.cuda(tmpweight)

    visnetwork.eval()

    vissavedir = f'./result-weightedCAM/{savename}/{all_or_t0}pairs/'

    if overwrite:
        os.removedirs(vissavedir)

    os.makedirs(f'{vissavedir}', exist_ok=True)
    if all_or_t0 == 'all':
        _visualize_all_pair(visnetwork, vissavedir, loader_test, subjidname)
    elif all_or_t0 == 't0':
        _visualize_t0_pair(visnetwork, vissavedir, loader_test, subjidname)
    else:
        assert "wrong choice of 'all_or_t0"
        
def visualize_crosssectional_regression(loader, opt, n_channels, subjidname, savename, overwrite=False):
    loader_test = torch.utils.data.DataLoader(
        loader(root=opt.image_dir, trainvaltest='test', transform=False, opt=opt),
        batch_size=1, shuffle=False, num_workers=opt.num_workers)

    savedmodelname = os.path.join(opt.save_name)
    loadmodel = Resnet18RegressionForPairVisFeature(channels=n_channels)
    visnetwork = load_single_model(loadmodel, savedmodelname)
    visnetwork.eval()
    vissavedir = f'./result-weightedCAM/{savename}/regression/'
    os.makedirs(f'{vissavedir}', exist_ok=True)
    _visualize_regression(visnetwork, vissavedir, loader_test, subjidname) # bothsign

    if overwrite:
        os.removedirs(vissavedir)

    os.makedirs(f'{vissavedir}', exist_ok=True)

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
parser.add_argument('--targetname', default='timepoint', type=str)


opt = parser.parse_args()
image_size = [int(item) for item in opt.image_size.split(',')]
opt.image_size = image_size



if __name__ == "__main__":

    # ## -- visualize PaIRNet
    visualize_PaIRNet(dict_dataloader[opt.dataname], opt, opt.image_channel, dict_subjectname[opt.dataname], opt.dataname, all_or_t0='all')

    # ## -- visualize crosssectional regression
    # visualize_crosssectional_regression(dict_dataloader[opt.dataname], opt, opt.image_channel, dict_subjectname[opt.dataname],  opt.datanam)


