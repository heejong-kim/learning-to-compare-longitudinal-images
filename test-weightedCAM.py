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

def load_pair_model(visnetwork):

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
            #
            # if 'features2.layer4.1.bn2' in f2:
            #     f22 = f2.replace('features2.layer4.1.bn2', 'features2.layer4.1.bn22')
            #     tmpweight[f22] = tmpweight[f2]
            #     print(f2, f22)
            #     del tmpweight[f2]

    if parallel:
        visnetwork = nn.DataParallel(visnetwork).to(device)
        visnetwork.load_state_dict(tmpweight)
    else:
        visnetwork.load_state_dict(tmpweight)
        if cuda:
            visnetwork = visnetwork.cuda(tmpweight)

    visnetwork.eval()
    model = visnetwork
    # bceloss = torch.nn.BCEWithLogitsLoss()
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
    # bceloss = torch.nn.BCEWithLogitsLoss()
    return model
def visualize_all_pair_featurebased(model, vissavedir, loader_test, subjidname):

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
        # print(len(model.module.gradients))
        model.module.release()
        # print(len(model.module.gradients))
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


    # TODO: revised this
    demo = loader_test.dataset.demo
    unqid = np.unique(demo[subjidname])
    pairindices = loader_test.dataset.index_combination.astype('int')
    count = 0
    subjectN = len(unqid)
    if subjectN > 150:
        unqid = unqid[:150]

    for uid in unqid:
        count += 1
        if count % int(subjectN/10) == 0 :
            print(f'{count} out of {subjectN}')

        vis_index = np.array(demo.index[demo[subjidname] == uid]).astype('int')

        if len(vis_index) >= 2:
            plt.close()
            plt.clf()
            fig, arr = plt.subplots(len(vis_index), len(vis_index), figsize=(30, 30))

            # todo change the code for the efficiency
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

                            # todo: 512*3*3 (weight the 512 with abs((f1-f2)*w)
                            _, activation1 = get_featurewise_gradcam(input1, model, 0)
                            _, activation2 = get_featurewise_gradcam(input2, model, 0)
                            # weight the activation
                            total_weight_reshape = np.repeat(np.repeat(total_weight[None, :, None, None], activation1.shape[-2], 2), activation1.shape[-1], 3)
                            weighted_activation1 = activation1 * total_weight_reshape
                            weighted_activation2 = activation2 * total_weight_reshape

                            # # TODO: try1: weight * activations
                            # features1 = {}
                            # features2 = {}
                            # # for fi in range(512):
                            # for fi in range(512): # grad, activations
                            #     features1[fi] = get_featurewise_gradcam(input1, model, fi)
                            #     features2[fi] = get_featurewise_gradcam(input2, model, fi)
                            #
                            #     if fi == 0:
                            #         weighted_activation1 = features1[fi][1] * total_weight[fi]
                            #         weighted_activation2 = features2[fi][1] * total_weight[fi]
                            #     else:
                            #         weighted_activation1 += features1[fi][1] * total_weight[fi]
                            #         weighted_activation2 = features2[fi][1] * total_weight[fi]

                            # # TODO: top 10 only
                            # featureidx = np.argsort(total_weight)[::-1][:10]
                            # # featureidx = np.argsort(total_weight_sign)[::-1][:10]
                            #
                            # weighted_activation1 = np.zeros(features1[fi][1].shape)
                            # weighted_activation2 = np.zeros(features2[fi][1].shape)
                            # for fi in featureidx: # grad, activations
                            #     if fi == 0:
                            #         weighted_activation1 = features1[0][1][:, fi,None, :] * total_weight[fi]
                            #         weighted_activation2 = features2[0][1][:, fi,None, :] * total_weight[fi]
                            #     else:
                            #         weighted_activation1 += features1[0][1][:, fi,None, :] * total_weight[fi]
                            #         weighted_activation2 = features1[0][1][:, fi,None, :] * total_weight[fi]

                            # TODO check input range always
                            cam1, inputvis1 = get_cam_image(weighted_activation1, input1)
                            cam2, inputvis2 = get_cam_image(weighted_activation2, input2)

                            # if inputvis2.shape[2] == 1:
                            #     inputvis1 = np.repeat(inputvis1, 3, 2)
                            #     inputvis2 = np.repeat(inputvis2, 3, 2)
                            #
                            # plt.imshow( np.concatenate((np.concatenate((cam1/255, cam2/255), 1), np.concatenate((inputvis1, inputvis2), 1)), 0))
                            # plt.savefig('./test-feature-oasis-top10.png') # embryo
                            # #
                            # # # concatenate 512 images
                            # # # for fi in range(512):
                            # plt.close(); plt.clf();
                            # fig, arr = plt.subplots(1, 1, figsize=(15, 6))
                            #
                            # for fi in featureidx:
                            #     # cam1, inputvis1 = get_bothsigncam_image(features1[fi][1][:, fi, None,:], features1[fi][0][:, fi, None,:],
                            #     #                                         np.minimum(np.maximum(input1, 0), 1))
                            #     # cam2, inputvis2 = get_bothsigncam_image(features2[fi][1][:, fi, None,:], features2[fi][0][:, fi, None,:],
                            #     #                                         np.minimum(np.maximum(input2, 0), 1))
                            #
                            #     cam1, inputvis1 = get_cam_image(features1[fi][1][:, fi, None,:],
                            #                                             np.minimum(np.maximum(input1, 0), 1))
                            #     cam2, inputvis2 = get_cam_image(features2[fi][1][:, fi, None,:],
                            #                                             np.minimum(np.maximum(input2, 0), 1))
                            #
                            #     if fi == featureidx[0] or fi == 0:
                            #         concatcol1 = cam1/255
                            #         concatcol2 = cam2/255
                            #         concatvis1 = inputvis1
                            #         concatvis2 = inputvis2
                            #     elif fi%32 < 32:
                            #         concatcol1 = np.concatenate((concatcol1, cam1/255), 1)
                            #         concatcol2 = np.concatenate((concatcol2, cam2/255), 1)
                            #         concatvis1 = np.concatenate((concatvis1, inputvis1), 1)
                            #         concatvis2 = np.concatenate((concatvis2, inputvis2), 1)
                            #
                            # if concatvis2.shape[2] == 1:
                            #     concatvis1 = np.repeat(concatvis1, 3, 2)
                            #     concatvis2 = np.repeat(concatvis2, 3, 2)
                            #
                            # plt.imshow( np.concatenate((concatcol1,    concatvis1), 0))
                            # plt.savefig('./test-feature1-oasis.png') # embryo
                            # plt.close(); plt.clf();
                            # fig, arr = plt.subplots(1, 1, figsize=(15, 6))
                            # plt.imshow( np.concatenate((concatcol2,    concatvis2), 0))
                            # plt.savefig('./test-feature2-oasis.png') # embryo

                            # cam1, cam2 = features1[fi][0], features2[fi][0]
                            # inputvis1, inputvis2 = features1[fi][0], features2[fi][0]
                            # np.concatneate()
                            #
                            # cam1, inputvis1 = get_bothsigncam_image(activations1, grad1, np.minimum(np.maximum(input1, 0), 1))
                            # cam2, inputvis2 = get_bothsigncam_image(activations2, grad2, np.minimum(np.maximum(input2, 0), 1))
                            #
                            # # cam1, inputvis1 = get_cam_image(activations1, grad1, input1)
                            # # cam2, inputvis2 = get_cam_image(activations2, grad2, input2)
                            #
                            # if inputvis1.shape[2] == 1:
                            #     inputvis1 = np.repeat(inputvis1, 3, 2)
                            #     inputvis2 = np.repeat(inputvis2, 3, 2)
                            if inputvis1.shape[2] == 1:
                                inputvis1 = np.repeat(inputvis1, 3, 2)
                                inputvis2 = np.repeat(inputvis2, 3, 2)

                            model.module.release()
                            output = np.sum(feature_weight * linear_weight)
                            arr[v][w].set_title(f'outputs:{output:.2f} \n target:{target1 - target2:.2f}')
                            arr[v][w].imshow(np.concatenate((np.concatenate((cam1/255, cam2/255), 1),
                                                            np.concatenate((inputvis1, inputvis2), 1)), 0))
                            # p = outputs.cpu().detach().numpy()[0]
                            # arr[v][w].set_title(f'O:{p[0]:.1f} / '
                            #                     f'T:{targetdiff.cpu().numpy()[0].astype("int")[0]:.1f} / '
                            #                     f'L:{loss.cpu().item():.1f}')
                        arr[v][w].axis('off')

            fig.savefig(f'{vissavedir}/{uid}.svg') # ./ layercam / fig-shapes-size
def visualize_t0_pair_featurebased(model, vissavedir, loader_test, subjidname):

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
        # print(len(model.module.gradients))
        model.module.release()
        # print(len(model.module.gradients))
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

    def get_cam_image_not_on_image(weighted_activations, input):
        targetsize = [input.shape[-2], input.shape[-1]]
        cam = weighted_activations.sum(axis=1)
        cam_min = np.min(cam)
        cam = cam - cam_min
        cam_max = np.max(cam)
        cam = cam / (1e-7 + cam_max)
        scaled = cv2.resize(cam[0], targetsize[::-1])

        inputvis = input.numpy().transpose(1, 2, 0)
        inputvis = np.minimum(np.maximum(inputvis, 0), 1)
        # cam_image = show_cam_on_image(inputvis,
        #                               scaled.squeeze(), use_rgb=True)
        return scaled.squeeze(), inputvis

    # TODO: revised this
    demo = loader_test.dataset.demo
    unqid = np.unique(demo[subjidname])
    pairindices = loader_test.dataset.index_combination.astype('int')
    count = 0
    subjectN = len(unqid)
    if subjectN > 150:
        unqid = unqid[:150]

    for uid in unqid:
        count += 1
        if count % int(subjectN/10) == 0 :
            print(f'{count} out of {subjectN}')

        vis_index = np.array(demo.index[demo[subjidname] == uid]).astype('int')
        if len(vis_index) >= 2:
            plt.close()
            plt.clf()
            fig, arr = plt.subplots(4, len(vis_index), figsize=(30, 12))

            # todo change the code for the efficiency
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
                total_weight_sign = (feature_weight*linear_weight).squeeze()


                # todo: 512*3*3 (weight the 512 with abs((f1-f2)*w)
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

                alpha = 0.7
                import matplotlib.colors as mcolors

                model.module.release()

# /                arr[0][count].imshow(inputvis1)
#                 arr[0][count].imshow(cam1, cmap="seismic",
#                                      norm=mcolors.TwoSlopeNorm(vmin=-1e-16, vmax=cam1.max(), vcenter=0),
#                                      alpha=alpha)
                output = np.sum(feature_weight * linear_weight)
                arr[0][w].set_title(f'outputs:{output:.2f} \n target:{target1-target2:.2f}')

                arr[0][w].imshow(cam1)
                arr[1][w].imshow(inputvis1)
                # arr[2][count].imshow(inputvis2)
                # arr[2][count].imshow(cam2, cmap="seismic",
                #                      norm=mcolors.TwoSlopeNorm(vmin=-1e-16, vmax=cam2.max(), vcenter=0),
                #                      alpha=alpha)
                arr[2][w].imshow(cam2)
                arr[3][w].imshow(inputvis2)
                arr[0][w].axis('off');
                arr[1][w].axis('off');
                arr[2][w].axis('off');
                arr[3][w].axis('off');

                # p = outputs.cpu().detach().numpy()[0]
                # arr[v][w].set_title(f'O:{p[0]:.1f} / '
                #                     f'T:{targetdiff.cpu().numpy()[0].astype("int")[0]:.1f} / '
                #                     f'L:{loss.cpu().item():.1f}')

            fig.savefig(f'{vissavedir}/{uid}.svg') # ./ layercam / fig-shapes-size
def visualize_regression_featurebased(model, vissavedir, loader_test, subjidname):

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
        # print(len(model.module.gradients))
        model.module.release()
        # print(len(model.module.gradients))
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

    demo = loader_test.dataset.demo
    unqid = np.unique(demo[subjidname])
    count = 0
    subjectN = len(unqid)
    if subjectN > 150:
        unqid = unqid[:150]
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
                arr[0][w].axis('off');
                arr[1][w].axis('off');

            fig.savefig(f'{vissavedir}/{uid}.svg') # ./ layercam / fig-shapes-size

def difference_model_vis_featurebased(loader, opt, n_channels, subjidname, savename, overwrite = False, all_or_t0 = 't0'): # bothsign
    loader_test = torch.utils.data.DataLoader(
        loader(root=opt.imagedir, trainvaltest='test', transform=False, opt=opt),
        batch_size=1, shuffle=False, num_workers=opt.num_workers)

    savedmodelname = os.path.join(opt.save_name, 'best.pth')
    tmpweight = torch.load(savedmodelname)
    # visnetwork = Resnet18DiffForPairVis(channels=n_channels)
    visnetwork = Resnet18DiffForPairVisFeature(channels=n_channels)

    if parallel:
        visnetwork = nn.DataParallel(visnetwork).to(device)
        visnetwork.load_state_dict(tmpweight)
    else:
        visnetwork.load_state_dict(tmpweight)
        if cuda:
            visnetwork = visnetwork.cuda(tmpweight)

    visnetwork.eval()

    vissavedir = f'./bothsigncam-featureweight/{savename}/{all_or_t0}pairs/'
    # vissavedir = f'./bothsigncam-featureweight-top10/{savename}/allpairs/'

    if overwrite:
        os.removedirs(vissavedir)

    os.makedirs(f'{vissavedir}', exist_ok=True)
    if all_or_t0 == 'all':
        visualize_all_pair_featurebased(visnetwork, vissavedir, loader_test, subjidname)
    elif all_or_t0 == 't0':
        visualize_t0_pair_featurebased(visnetwork, vissavedir, loader_test, subjidname)
    else:
        assert "wrong choice of 'all_or_t0"
def regression_model_vis_featurebased(loader, opt, n_channels, subjidname, savename, overwrite=False):
    loader_test = torch.utils.data.DataLoader(
        loader(root=opt.imagedir, trainvaltest='test', transform=False, opt=opt),
        batch_size=1, shuffle=False, num_workers=opt.num_workers)

    savedmodelname = os.path.join(opt.save_name, 'best.pth')
    loadmodel = Resnet18RegressionForPairVisFeature(channels=n_channels)
    visnetwork = load_single_model(loadmodel, savedmodelname)
    visnetwork.eval()
    vissavedir = f'./bothsigncam-featureweight/{savename}/regression/'
    os.makedirs(f'{vissavedir}', exist_ok=True)
    visualize_regression_featurebased(visnetwork, vissavedir, loader_test, subjidname) # bothsign

    if overwrite:
        os.removedirs(vissavedir)

    os.makedirs(f'{vissavedir}', exist_ok=True)


unqid_dic = {'embryo':'embryoname', 'tumor': 'Subject ID', 'starmen': 'id','oasis': 'subject-id'}

cuda = True
parallel = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Compare with actual time difference
parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--image_size', default="68,68", type=str, help="x,y", required=True)
parser.add_argument('--image_channel', default=1, type=int)
parser.add_argument('--image_dir', default='./datasets/starmen-augmentation', type=str)
parser.add_argument('--dataname', type=str, required=True)
parser.add_argument('--save_name', type=str, required=True)

opt = parser.parse_args()
set_manual_seed(opt.seed)
suffix = f'seed{torch.initial_seed()}'

image_size = [int(item) for item in opt.image_size.split(',')]
opt.image_size = image_size


## MIDL result ------------------------------------------------------

# ---- starmen augmentation pair
dataname = 'starmen-augmentation'
# dataname = 'starmen'
savename = dataname
expname = ''
opt.imagesize = [68, 68]
opt.imagedir = f'/scratch/datasets/hk672/{dataname}'
opt.targetname = 'timepoint'
loader = STARMEN #FIGURES
n_channels = 1
subjidname = 'id'
task = 'diff-resnet18-1layer'
params = f'lr{best_lr_dic[task][savename]}-b10.9-b20.999seed0/{expname}'
opt.save_name = 'saved_models/' +  f'starmen-augmentation/{params}/{task}/'
# difference_model_vis_bothsigncam(loader, opt, n_channels, subjidname, savename)
difference_model_vis_featurebased(loader, opt, n_channels, subjidname, 'starmen-augmentation',all_or_t0='all')


# -- delta reg
dataname = 'starmen-augmentation'
# dataname = 'starmen'
savename = dataname
expname = ''
opt.imagesize = [68, 68]
opt.imagedir = f'/scratch/datasets/hk672/{dataname}'
opt.targetname = 'timepoint'
loader = STARMEN #FIGURES
n_channels = 1
subjidname = 'id'
task = 'resnet18-1layer-deltareg-supervised'
params = f'lr{best_lr_dic[task][savename]}-b10.9-b20.999seed0/{expname}'
opt.save_name = 'saved_models/' +  f'starmen-augmentation/{params}/{task}/'
# difference_model_vis_bothsigncam(loader, opt, n_channels, subjidname, savename)
difference_model_vis_featurebased(loader, opt, n_channels, subjidname, 'starmen-augmentation-deltareg',all_or_t0='all')


# ---- starmen augmentation regression  TODO: RE
dataname = 'starmen-augmentation'
# dataname = 'starmen'
savename = dataname
expname = ''
opt.imagesize = [68, 68]
opt.imagedir = f'/scratch/datasets/hk672/{dataname}'
opt.targetname = 't_star'
loader = STARMENregression #FIGURES
n_channels = 1
subjidname = 'id'
task = 'resnet18-1layer-regression-supervised'
params = f'lr{best_lr_dic[task][savename]}-b10.9-b20.999seed0/{expname}'
opt.save_name = 'saved_models/' +  f'starmen-augmentation/{params}/{task}/'
# regression_model_vis_bothsigncam(loader, opt, n_channels, subjidname, savename)
regression_model_vis_featurebased(loader, opt, n_channels, subjidname,  'starmen-augmentation')


# ---- synthetic tumor pair
dataname = 'oasis-tumor'
savename = 'tumor'
dataname = 'oasis-tumor-wo-preprocess'
savename = 'tumor-wo-preprocess'
expname = ''
opt.imagesize = [200, 200]
opt.imagedir = f'/scratch/datasets/hk672/{dataname}'
opt.targetname = 'radius'
loader = TUMOR #FIGURES
n_channels = 3
subjidname = 'Subject ID'
task = 'diff-resnet18-1layer'
params = f'lr{best_lr_dic[task][savename]}-b10.9-b20.999seed0/{expname}'
opt.save_name = 'saved_models/' +  f'{savename}/{params}/{task}/'
difference_model_vis_featurebased(loader, opt, n_channels, subjidname, savename,all_or_t0='all')

# -- delta reg
dataname = 'oasis-tumor'
savename = 'tumor'
dataname = 'oasis-tumor-wo-preprocess'
savename = 'tumor-wo-preprocess'
expname = ''
opt.imagesize = [200, 200]
opt.imagedir = f'/scratch/datasets/hk672/{dataname}'
opt.targetname = 'radius'
loader = TUMOR #FIGURES
n_channels = 3
subjidname = 'Subject ID'
task = 'resnet18-1layer-deltareg-supervised'
params = f'lr{best_lr_dic[task][savename]}-b10.9-b20.999seed0/{expname}'
opt.save_name = 'saved_models/' +  f'{savename}/{params}/{task}/'
# difference_model_vis_featurebased(loader, opt, n_channels, subjidname, 'tumor-deltareg',all_or_t0='all')
difference_model_vis_featurebased(loader, opt, n_channels, subjidname, 'tumor-wo-process-deltareg',all_or_t0='all')

# difference_model_vis_featurebased(loader, opt, n_channels, subjidname, savename,all_or_t0='t0')


# ---- synthetic tumor regression # todo ** rerunning (overwritten)
dataname = 'oasis-tumor'
savename = 'tumor'
dataname = 'oasis-tumor-wo-preprocess'
savename = 'tumor-wo-preprocess'
expname = ''
opt.imagesize = [200, 200]
opt.imagedir = f'/scratch/datasets/hk672/{dataname}'
opt.targetname = 'radius'
loader = TUMORregression #FIGURES
n_channels = 3
subjidname = 'Subject ID'
task = 'resnet18-1layer-regression-supervised'
params = f'lr{best_lr_dic[task][savename]}-b10.9-b20.999seed0/{expname}'
opt.save_name = 'saved_models/' +  f'{savename}/{params}/{task}/'
# regression_model_vis_bothsigncam(loader, opt, n_channels, subjidname, savename)
regression_model_vis_featurebased(loader, opt, n_channels, subjidname, savename)


# ---- embryo pair
dataname = 'embryo'
savename = dataname
expname = ''
opt.imagesize = [200, 200]
opt.imagedir = f'/scratch/datasets/hk672/{dataname}'
opt.targetname = 'phaseidx'
loader = EMBRYO #FIGURES
n_channels = 1
subjidname = 'embryoidx'
task = 'diff-resnet18-1layer'
params = f'lr{best_lr_dic[task][savename]}-b10.9-b20.999seed0/{expname}'
opt.save_name = 'saved_models/' +  f'{savename}/{params}/{task}/'
difference_model_vis_featurebased(loader, opt, n_channels, subjidname, savename,all_or_t0='all')
# difference_model_vis_featurebased(loader, opt, n_channels, subjidname, savename,all_or_t0='t0')

# --- deltareg
dataname = 'embryo'
savename = dataname
expname = ''
opt.imagesize = [200, 200]
opt.imagedir = f'/scratch/datasets/hk672/{dataname}'
opt.targetname = 'phaseidx'
loader = EMBRYO #FIGURES
n_channels = 1
subjidname = 'embryoidx'
task = 'resnet18-1layer-deltareg-supervised'
params = f'lr{best_lr_dic[task][savename]}-b10.9-b20.999seed0/{expname}'
opt.save_name = 'saved_models/' +  f'{savename}/{params}/{task}/'
difference_model_vis_featurebased(loader, opt, n_channels, subjidname, savename+'-deltareg',all_or_t0='all')
# difference_model_vis_featurebased(loader, opt, n_channels, subjidname, savename,all_or_t0='t0')


# ---- embryo pair regression
dataname = 'embryo'
savename = dataname
expname = ''
opt.imagesize = [200, 200]
opt.imagedir = f'/scratch/datasets/hk672/{dataname}'
opt.targetname = 'phaseidx'
loader = EMBRYOregression #FIGURES
n_channels = 1
subjidname = 'embryoidx'
task = 'resnet18-1layer-regression-supervised'
params = f'lr{best_lr_dic[task][savename]}-b10.9-b20.999seed0/{expname}'
opt.save_name = 'saved_models/' +  f'{savename}/{params}/{task}/'
# regression_model_vis_bothsigncam(loader, opt, n_channels, subjidname, savename)
regression_model_vis_featurebased(loader, opt, n_channels, subjidname, savename)



# ---- oasis aging pair
dataname = 'oasis'
savename = 'brainaging'
expname = ''
opt.imagesize = [176, 256]
opt.imagedir = f'/scratch/datasets/hk672/{dataname}'
opt.targetname = 'age'
loader = OASIS #FIGURES
n_channels = 1
subjidname = 'subject-id'
task = 'diff-resnet18-1layer'
params = f'lr{best_lr_dic[task][savename]}-b10.9-b20.999seed0/{expname}'
opt.save_name = 'saved_models/' +  f'{savename}/{params}/{task}/'
# difference_model_vis_bothsigncam(loader, opt, n_channels, subjidname, savename)
difference_model_vis_featurebased(loader, opt, n_channels, subjidname, savename,all_or_t0='all')
# difference_model_vis_featurebased(loader, opt, n_channels, subjidname, savename,all_or_t0='t0')

# --- deltareg
dataname = 'oasis'
savename = 'brainaging'
expname = ''
opt.imagesize = [176, 256]
opt.imagedir = f'/scratch/datasets/hk672/{dataname}'
opt.targetname = 'age'
loader = OASIS #FIGURES
n_channels = 1
subjidname = 'subject-id'
task = 'resnet18-1layer-deltareg-supervised'
params = f'lr{best_lr_dic[task][savename]}-b10.9-b20.999seed0/{expname}'
opt.save_name = 'saved_models/' +  f'{savename}/{params}/{task}/'
# difference_model_vis_bothsigncam(loader, opt, n_channels, subjidname, savename)
difference_model_vis_featurebased(loader, opt, n_channels, subjidname, savename+'-deltareg',all_or_t0='all')
# difference_model_vis_featurebased(loader, opt, n_channels, subjidname, savename,all_or_t0='t0')

# ---- oasis aging regression
dataname = 'oasis'
savename = 'brainaging'
expname = ''
opt.imagesize = [176, 256]
opt.imagedir = f'/scratch/datasets/hk672/{dataname}'
opt.targetname = 'age'
loader = OASISregression #FIGURES
n_channels = 1
subjidname = 'subject-id'
task = 'resnet18-1layer-regression-supervised'
params = f'lr{best_lr_dic[task][savename]}-b10.9-b20.999seed0/{expname}'
opt.save_name = 'saved_models/' +  f'{savename}/{params}/{task}/'
# regression_model_vis_bothsigncam(loader, opt, n_channels, subjidname, savename)
regression_model_vis_featurebased(loader, opt, n_channels, subjidname, savename)

# --------------------------------------------------------------------------------------------

# ---- synthetic tumors pair
dataname = 'oasis-tumors'
savename = 'tumors'
expname = ''
params = f'lr0.01-b10.9-b20.999seed0/{expname}'
opt.imagesize = [200, 200]
opt.imagedir = f'/scratch/datasets/hk672/{dataname}'
opt.targetname = 'radius'
loader = TUMOR #FIGURES
n_channels = 3
subjidname = 'Subject ID'
task = 'diff-resnet18-1layer'
opt.save_name = 'saved_models/' +  f'{savename}/{params}/{task}/'
difference_model_vis_bothsigncam(loader, opt, n_channels, subjidname, savename)


# ---- synthetic tumors regression
dataname = 'oasis-tumors'
savename = 'tumors'
expname = ''
params = f'lr0.0001-b10.9-b20.999seed0/{expname}' # 0001 or 001
opt.imagesize = [200, 200]
opt.imagedir = f'/scratch/datasets/hk672/{dataname}'
opt.targetname = 'radius'
loader = TUMORregression #FIGURES
n_channels = 3
subjidname = 'Subject ID'
task = 'resnet18-1layer-regression-supervised'
opt.save_name = 'saved_models/' +  f'{savename}/{params}/{task}/'
regression_model_vis_bothsigncam(loader, opt, n_channels, subjidname, savename)

# ---- clockface pair
dataname = 'clockface-similarsize-shapes'
savename = dataname
expname = 'randomtimepoints'
params = f'lr0.001-b10.9-b20.999seed0/{expname}'
opt.imagesize = [200, 200]
opt.imagedir = f'/scratch/datasets/hk672/{dataname}'
opt.targetname = 'time'
loader = CLOCKFACE #FIGURES
n_channels = 3
subjidname = 'id'
task = 'diff-resnet18-1layer'
opt.save_name = 'saved_models/' +  f'{savename}/{params}/{task}/'
# difference_model_vis_bothsigncam(loader, opt, n_channels, subjidname, savename)
difference_model_vis_featurebased(loader, opt, n_channels, subjidname, savename)

# ---- clockface notransform pair
dataname = 'clockface-similarsize-shapes'
savename = dataname
expname = 'randomtimepoints'
params = f'lr0.001-b10.9-b20.999seed0-notransform/{expname}'
opt.imagesize = [200, 200]
opt.imagedir = f'/scratch/datasets/hk672/{dataname}'
opt.targetname = 'time'
loader = CLOCKFACE #FIGURES
n_channels = 3
subjidname = 'id'
task = 'diff-resnet18-1layer'
opt.save_name = 'saved_models/' +  f'{savename}/{params}/{task}/'
# difference_model_vis_bothsigncam(loader, opt, n_channels, subjidname, savename+ '-notransform')
difference_model_vis_featurebased(loader, opt, n_channels, subjidname, savename+ '-notransform')


# ---- clockface regression
dataname = 'clockface-similarsize-shapes'
savename = dataname
expname = 'randomtimepoints'
params = f'lr0.001-b10.9-b20.999seed0/{expname}'
opt.imagesize = [200, 200]
opt.imagedir = f'/scratch/datasets/hk672/{dataname}'
opt.targetname = 'time'
loader = CLOCKFACEregression #FIGURES
n_channels = 3
subjidname = 'id'
task = 'resnet18-1layer-regression-supervised'
opt.save_name = 'saved_models/' +  f'{savename}/{params}/{task}/'
regression_model_vis_bothsigncam(loader, opt, n_channels, subjidname, savename)


# ---- clockface notransform regression
dataname = 'clockface-similarsize-shapes'
savename = dataname
expname = 'randomtimepoints'
params = f'lr0.001-b10.9-b20.999seed0-notransform/{expname}'
opt.imagesize = [200, 200]
opt.imagedir = f'/scratch/datasets/hk672/{dataname}'
opt.targetname = 'time'
loader = CLOCKFACEregression #FIGURES
n_channels = 3
subjidname = 'id'
task = 'resnet18-1layer-regression-supervised'
opt.save_name = 'saved_models/' +  f'{savename}/{params}/{task}/'
regression_model_vis_bothsigncam(loader, opt, n_channels, subjidname, savename + '-notransform')



# ---- starmen pair
dataname = 'starmen'
savename = dataname
expname = ''
params = f'lr0.001-b10.9-b20.999seed0/{expname}'
opt.imagesize = [68, 68]
opt.imagedir = f'/scratch/datasets/hk672/{dataname}'
opt.targetname = 'timepoint'
loader = STARMEN #FIGURES
n_channels = 1
subjidname = 'id'
task = 'diff-resnet18-1layer'
opt.save_name = 'saved_models/' +  f'{savename}/{params}/{task}/'
# difference_model_vis_bothsigncam(loader, opt, n_channels, subjidname, savename)
difference_model_vis_featurebased(loader, opt, n_channels, subjidname, savename)

# ---- starmen augmentation pair
dataname = 'starmen-augmentation'
dataname = 'starmen'
savename = dataname
expname = ''
params = f'lr0.001-b10.9-b20.999seed0/{expname}'
opt.imagesize = [68, 68]
opt.imagedir = f'/scratch/datasets/hk672/{dataname}'
opt.targetname = 'timepoint'
loader = STARMEN #FIGURES
n_channels = 1
subjidname = 'id'
task = 'diff-resnet18-1layer'
opt.save_name = 'saved_models/' +  f'starmen-augmentation/{params}/{task}/'
# difference_model_vis_bothsigncam(loader, opt, n_channels, subjidname, savename)
difference_model_vis_featurebased(loader, opt, n_channels, subjidname, 'starmen-augmentation')




# ---- starmen augmentation pair
dataname = 'starmen-augmentation-additional'
# dataname = 'starmen'
savename = dataname
expname = ''
opt.imagesize = [68, 68]
opt.imagedir = f'./data/{dataname}'
opt.targetname = 'timepoint'
loader = STARMEN #FIGURES
n_channels = 1
subjidname = 'id'
task = 'diff-resnet18-1layer'
params = f'lr{best_lr_dic[task]["starmen-augmentation"]}-b10.9-b20.999seed0/{expname}'
opt.save_name = 'saved_models/' +  f'starmen-augmentation/{params}/{task}/'
# difference_model_vis_bothsigncam(loader, opt, n_channels, subjidname, savename)
difference_model_vis_featurebased(loader, opt, n_channels, subjidname, 'starmen-augmentation-additional',all_or_t0='t0')

