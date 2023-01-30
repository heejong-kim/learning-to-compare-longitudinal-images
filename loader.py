import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import cv2
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import torch

class STARMEN(Dataset):
    def __init__(self, root='./data/starmen', transform=None, trainvaltest='train', opt = None):

        self.trainvaltest = trainvaltest
        self.imgdir = os.path.join(root)

        if 'demoname' in opt:
            meta = pd.read_csv(os.path.join(root, opt.demoname), index_col=0)
        else:
            meta = pd.read_csv(os.path.join(root, 'df.csv'), index_col=0)

        meta = meta[meta.trainvaltest == trainvaltest].reset_index()

        IDunq = np.unique(meta.id)
        index_combination = np.empty((0, 2))
        for sid in IDunq:
            indices = np.where(meta.id == sid)[0]
            ### all possible pairs
            tmp_combination = np.array(
                np.meshgrid(np.array(range(len(indices))), np.array(range(len(indices))))).T.reshape(-1, 2)

            index_combination = np.append(index_combination, indices[tmp_combination], 0)

        if opt == None:
            img_height, img_width = [200, 200]
        else:
            img_height, img_width = opt.imagesize
            self.targetname = opt.targetname

        self.resize = transforms.Compose([
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ToTensor(),
        ])

        self.index_combination = index_combination
        self.transform = transform
        self.demo = meta


    def __getitem__(self, index):
        index1, index2 = self.index_combination[index]
        target1, target2 = self.demo[self.targetname][index1], self.demo[self.targetname][index2]
        img1 = np.load(os.path.join(self.imgdir, self.demo.path[index1]))
        img1 = self.resize(Image.fromarray(img1)) 
        img2 = np.load(os.path.join(self.imgdir, self.demo.path[index2]))
        img2 = self.resize(Image.fromarray(img2)) 

        if self.transform:
            augmentation = transforms.Compose([
                transforms.RandomApply(torch.nn.ModuleList(
                    [transforms.RandomAffine(degrees=(0, 0), scale=(0.5, 1.0),
                                             interpolation=InterpolationMode.BILINEAR)]),
                    p=0.5),
            ])

            img1 = augmentation(img1)
            img2 = augmentation(img2)

        return [np.array(img1), target1], [np.array(img2), target2]

    def __len__(self):
        return len(self.index_combination)

class STARMENregression(Dataset):
    def __init__(self, root='./data/starmen', transform=None, trainvaltest='train', opt = None):

        self.trainvaltest = trainvaltest
        self.imgdir = os.path.join(root)

        if 'demoname' in opt:
            meta = pd.read_csv(os.path.join(root, opt.demoname), index_col=0)
        else:
            meta = pd.read_csv(os.path.join(root, 'df.csv'), index_col=0)

        meta = meta[meta.trainvaltest == trainvaltest].reset_index()

        if opt == None:
            img_height, img_width = [200, 200]
        else:
            img_height, img_width = opt.imagesize
            self.targetname = opt.targetname

        self.resize = transforms.Compose([
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ToTensor(),
        ])

        self.transform = transform
        self.demo = meta

    def __getitem__(self, index):
        target = self.demo[self.targetname][index]
        img = np.load(os.path.join(self.imgdir, self.demo.path[index]))
        img = self.resize(Image.fromarray(img)) 

        if self.transform:
            augmentation = transforms.Compose([
                transforms.RandomApply(torch.nn.ModuleList(
                    [transforms.RandomAffine(degrees=(0, 0), scale=(0.5, 1.0),
                                             interpolation=InterpolationMode.BILINEAR)]),
                    p=0.5),
            ])

            img = augmentation(img)

        return np.array(img), target

    def __len__(self):
        return len(self.demo)

class TUMOR(Dataset):
    def __init__(self, root='./data/oasis-tumor', transform=None, trainvaltest='train', opt = None):

        self.trainvaltest = trainvaltest
        self.imgdir = os.path.join(root) + '/images'
        meta = pd.read_csv(os.path.join(root, 'demo-oasis-synthetic-tumor.csv'), index_col=0)
        self.targetname = opt.targetname

        meta = meta[meta.trainvaltest == trainvaltest].reset_index()
        IDunq = np.unique(meta['Subject ID'])
        index_combination = np.empty((0, 2))
        for sid in IDunq:
            indices = np.where(meta['Subject ID'] == sid)[0]
            ### all possible pairs
            tmp_combination = np.array(
                np.meshgrid(np.array(range(len(indices))), np.array(range(len(indices))))).T.reshape(-1, 2)
            index_combination = np.append(index_combination, indices[tmp_combination], 0)

        if opt == None:
            img_height, img_width = [200, 200]
        else:
            img_height, img_width = opt.imagesize

        self.resize = transforms.Compose([
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ToTensor(),
        ])

        self.index_combination = index_combination
        self.transform = transform
        self.demo = meta

    def __getitem__(self, index):
        index1, index2 = self.index_combination[index]
        target1, target2 = self.demo[self.targetname][index1], self.demo[self.targetname][index2]
        img1 = Image.open(os.path.join(self.imgdir, self.demo.fname[index1]))
        img1 = self.resize(img1) 
        img2 = Image.open(os.path.join(self.imgdir, self.demo.fname[index2]))
        img2 = self.resize(img2) 

        if self.transform:
            augmentation = transforms.Compose([
                transforms.RandomApply(torch.nn.ModuleList(
                    [transforms.RandomAffine(degrees=(-10, 10), translate=(0.05,0.05),
                                             interpolation=InterpolationMode.BILINEAR)]),
                    p=0.5),
            ])

            img1 = augmentation(img1)
            img2 = augmentation(img2)

        return [np.array(img1), target1], [np.array(img2), target2]

    def __len__(self):
        return len(self.index_combination)

class TUMORregression(Dataset):
    def __init__(self, root='./data/oasis-tumor', transform=None, trainvaltest='train', opt = None):

        self.trainvaltest = trainvaltest
        self.imgdir = os.path.join(root) + '/images'
        meta = pd.read_csv(os.path.join(root, 'demo-oasis-synthetic-tumor.csv'), index_col=0)
        self.targetname = opt.targetname

        meta = meta[meta.trainvaltest == trainvaltest].reset_index()

        if opt == None:
            img_height, img_width = [200, 200]
        else:
            img_height, img_width = opt.imagesize

        self.resize = transforms.Compose([
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ToTensor(),
        ])

        self.transform = transform
        self.demo = meta

    def __getitem__(self, index):
        target = self.demo[self.targetname][index]
        img = Image.open(os.path.join(self.imgdir, self.demo.fname[index]))
        img = self.resize(img) 

        if self.transform:
            augmentation = transforms.Compose([
                transforms.RandomApply(torch.nn.ModuleList(
                    [transforms.RandomAffine(degrees=(-10, 10), translate=(0.05,0.05),
                                             interpolation=InterpolationMode.BILINEAR)]),
                    p=0.5),
            ])

            img = augmentation(img)

        return np.array(img), target

    def __len__(self):
        return len(self.demo)

class OASIS(Dataset):
    def __init__(self, root='/nfs04/data/OASIS3/aligned-midslice', transform=None, trainvaltest='train', opt = None):

        self.trainvaltest = trainvaltest
        self.imgdir = os.path.join(root, 'images/')
        self.targetname = opt.targetname

        if 'demoname' in opt:
            meta = pd.read_csv(os.path.join(root, opt.demoname), index_col=0)
        else:
            meta = pd.read_csv(os.path.join(root, 'demo-healthy-longitudinal.csv'), index_col=0)

        meta = meta[meta.trainvaltest == trainvaltest].reset_index()
        IDunq = np.unique(meta['subject-id'])
        index_combination = np.empty((0, 2))
        for sid in IDunq:
            indices = np.where(meta['subject-id'] == sid)[0]
            ### all possible pairs
            tmp_combination = np.array(
                np.meshgrid(np.array(range(len(indices))), np.array(range(len(indices))))).T.reshape(-1, 2)
            index_combination = np.append(index_combination, indices[tmp_combination], 0)

        if opt == None:
            img_height, img_width = [176, 256]
        else:
            img_height, img_width = opt.imagesize

        self.resize = transforms.Compose([
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ToTensor(),
        ])

        self.index_combination = index_combination
        self.transform = transform
        self.demo = meta

    def __getitem__(self, index):
        index1, index2 = self.index_combination[index]
        target1, target2 = self.demo[self.targetname][index1], self.demo[self.targetname][index2]
        img1 = Image.open(os.path.join(self.imgdir, self.demo.fname[index1]))
        img1 = self.resize(img1) 
        img2 = Image.open(os.path.join(self.imgdir, self.demo.fname[index2]))
        img2 = self.resize(img2) 

        if self.transform:
            augmentation = transforms.Compose([
                transforms.RandomApply(torch.nn.ModuleList(
                    [transforms.RandomAffine(degrees=(-10, 10), translate=(0.05,0.05), #scale=(0.9, 1.1),
                                             interpolation=InterpolationMode.BILINEAR)]),
                    p=0.5),
            ])

            img1 = augmentation(img1)
            img2 = augmentation(img2)

        return [np.array(img1), target1], [np.array(img2), target2]

    def __len__(self):
        return len(self.index_combination)

class OASISregression(Dataset):
    def __init__(self, root='/nfs04/data/OASIS3/aligned-midslice', transform=None, trainvaltest='train', opt = None):

        self.trainvaltest = trainvaltest
        self.imgdir = os.path.join(root, 'images/')
        self.targetname = opt.targetname

        if 'demoname' in opt:
            meta = pd.read_csv(os.path.join(root, opt.demoname), index_col=0)
        else:
            meta = pd.read_csv(os.path.join(root, 'demo-healthy-longitudinal.csv'), index_col=0)


        meta = meta[meta.trainvaltest == trainvaltest].reset_index()

        if opt == None:
            img_height, img_width = [176, 256]
        else:
            img_height, img_width = opt.imagesize

        self.resize = transforms.Compose([
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ToTensor(),
        ])

        self.transform = transform
        self.demo = meta

    def __getitem__(self, index):
        target = self.demo[self.targetname][index]
        img = Image.open(os.path.join(self.imgdir, self.demo.fname[index]))
        img = self.resize(img) 

        if self.transform:
            augmentation = transforms.Compose([
                transforms.RandomApply(torch.nn.ModuleList(
                    [transforms.RandomAffine(degrees=(-10, 10), translate=(0.05,0.05),#scale=(0.9, 1.1),
                                             interpolation=InterpolationMode.BILINEAR)]),
                    p=0.5),
            ])

            img = augmentation(img)

        return np.array(img), target


    def __len__(self):
        return len(self.demo)

class EMBRYO(Dataset):

    def __init__(self, root='./data/embryo/', transform=None, trainvaltest='train', opt = None):
        self.trainvaltest = trainvaltest
        self.imgdir = os.path.join(root, 'embryo_dataset')

        meta = pd.read_csv(os.path.join(root, 'demo.csv'), index_col=0)

        num_of_subjects = len(np.unique(meta['embryoname']))
        meta = meta[meta.trainvaltest == trainvaltest].reset_index()
        IDunq = np.unique(meta['embryoname'])
        index_combination = np.empty((0, 2))
        for sid in IDunq:
            indices = np.where(meta['embryoname'] == sid)[0]
            ### all possible pairs
            tmp_combination = np.array(
                np.meshgrid(np.array(range(len(indices))), np.array(range(len(indices))))).T.reshape(-1, 2)
            # ### permutation pairs
            # tmp_combination = np.concatenate((np.random.permutation(range(len(indices)))[:, None],
            #                                   np.random.permutation(range(len(indices)))[:, None]), 1)
            index_combination = np.append(index_combination, indices[tmp_combination], 0)

        if opt == None:
            img_height, img_width = [500, 500]
            self.targetname = 'phaseidx'
        else:
            img_height, img_width = opt.imagesize
            self.targetname = opt.targetname

        self.resize = transforms.Compose([
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ToTensor(),
        ])

        self.index_combination = index_combination
        self.transform = transform
        self.demo = meta

    def __getitem__(self, index):
        index1, index2 = self.index_combination[index]
        target1, target2 = self.demo[self.targetname][index1], self.demo[self.targetname][index2]
        img1 = Image.open(os.path.join(self.imgdir, self.demo.fname[index1]))
        img1 = self.resize(img1) 
        img2 = Image.open(os.path.join(self.imgdir, self.demo.fname[index2]))
        img2 = self.resize(img2) 

        if self.transform:
            augmentation = transforms.Compose([
                transforms.RandomApply(torch.nn.ModuleList(
                    [transforms.RandomAffine(degrees=(-10, 10), translate=(0.05,0.05),
                                             interpolation=InterpolationMode.BILINEAR)]),
                    p=0.5),
            ])

            img1 = augmentation(img1)
            img2 = augmentation(img2)

        return [np.array(img1), target1], [np.array(img2), target2]

    def __len__(self):
        return len(self.index_combination)

class EMBRYOregression(Dataset):

    def __init__(self, root='./data/embryo/embryo_dataset/', transform=None, trainvaltest='train', opt = None):
        self.trainvaltest = trainvaltest
        self.imgdir = os.path.join(root, 'embryo_dataset')
        meta = pd.read_csv(os.path.join(root, 'demo.csv'), index_col=0)

        num_of_subjects = len(np.unique(meta['embryoname']))
        meta = meta[meta.trainvaltest == trainvaltest].reset_index()

        if opt == None:
            img_height, img_width = [500, 500]
            self.targetname = 'phaseidx'
        else:
            img_height, img_width = opt.imagesize
            self.targetname = opt.targetname


        self.resize = transforms.Compose([
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ToTensor(),
        ])

        self.transform = transform
        self.demo = meta

    def __getitem__(self, index):
        target = self.demo[self.targetname][index]
        img = Image.open(os.path.join(self.imgdir, self.demo.fname[index]))
        img = self.resize(img) 

        if self.transform:
            augmentation = transforms.Compose([
                transforms.RandomApply(torch.nn.ModuleList(
                    [transforms.RandomAffine(degrees=(-10, 10), translate=(0.05,0.05),
                                             interpolation=InterpolationMode.BILINEAR)]),
                    p=0.5),
            ])

            img = augmentation(img)

        return np.array(img), target

    def __len__(self):
        return len(self.demo)
    
