import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import cv2
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import torch
import torchio as tio
import nibabel as nib

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
            img_height, img_width = opt.image_size
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
            img_height, img_width = opt.image_size
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
            img_height, img_width = opt.image_size

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
            img_height, img_width = opt.image_size

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
            img_height, img_width = opt.image_size

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
            img_height, img_width = opt.image_size

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
            img_height, img_width = opt.image_size
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
            img_height, img_width = opt.image_size
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
    
class OASIS3D(Dataset):
    def __init__(self, root='/share/sablab/nfs04/data/OASIS3/', transform=None, trainvaltest='train', opt = None):

        self.trainvaltest = trainvaltest
        self.imgdir = os.path.join(root, 'image/')
        self.targetname = opt.targetname

        if 'demoname' in opt:
            meta = pd.read_csv(os.path.join(root, opt.demoname), index_col=0)
        else:
            meta = pd.read_csv(os.path.join(root, 'demo/demo-healthy-longitudinal-3D.csv'), index_col=0)

        # fname = np.loadtxt('/share/sablab/nfs04/data/OASIS3/affine-alignment/imagelist.csv', 'str')
        # meta.fname = fname
        # # matching name test
        # fname = meta.fname.str.split('/', expand=True)[8]
        # fnametmp = fname.str.split('_', expand=True)
        # np.all(fnametmp[1] == meta['session-id'])
        # meta.to_csv(os.path.join(root, 'demo/demo-healthy-longitudinal-3D.csv'))

        meta = meta[meta.trainvaltest == trainvaltest].reset_index()
        IDunq = np.unique(meta['subject-id'])
        index_combination = np.empty((0, 2))
        for sid in IDunq:
            indices = np.where(meta['subject-id'] == sid)[0]
            ### all possible pairs
            tmp_combination = np.array(
                np.meshgrid(np.array(range(len(indices))), np.array(range(len(indices))))).T.reshape(-1, 2)
            index_combination = np.append(index_combination, indices[tmp_combination], 0)

        if transform:
            transforms = []
            affine = tio.RandomAffine(scales=0.1,
                                         degrees=10,
                                         translation=5,
                                         image_interpolation='linear',
                                         default_pad_value='otsu')  # bspline
            transforms.append(affine)
            intensity = tio.OneOf({
                # tio.RandomBiasField(): 0.5,
                tio.RandomNoise(): 0.5,
                tio.RandomBlur(): 0.5
            },
                p=0.5,
            )
            transforms.append(intensity)
            self.transform = tio.Compose(transforms)
        else:
            self.transform=False

        self.index_combination = index_combination
        self.demo = meta
        self.image_size = opt.image_size

    def __getitem__(self, index):
        index1, index2 = self.index_combination[index]
        target1, target2 = self.demo[self.targetname][index1], self.demo[self.targetname][index2]

        image1 = tio.Subject(
            t1=tio.ScalarImage(os.path.join(self.imgdir, self.demo.fname[index1])),
            label=[],
            diagnosis='',
        )
        image2 = tio.Subject(
            t1=tio.ScalarImage(os.path.join(self.imgdir, self.demo.fname[index2])),
            label=[],
            diagnosis='',
        )

        if image1.shape[1:] != tuple(self.image_size):
            resize = tio.transforms.Resize(tuple(self.image_size))
            image1 = resize(image1)
            image2 = resize(image2)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)


        return [np.array(image1)[0], target1], [np.array(image2)[0], target2]

    def __len__(self):
        return len(self.index_combination)
