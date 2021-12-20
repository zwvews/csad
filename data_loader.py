# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os


import torch.utils.data as data
import torchvision.datasets as datasets


class WholeDataLoader(Dataset):
    def __init__(self,option, istrain=True):
        # self.data_split = istrain
        self.is_train = istrain
        data_dic = np.load(os.path.join(option.data_dir,'mnist_10color_jitter_var_%.03f.npy'%option.color_var),encoding='latin1', allow_pickle=True).item()
        # print(os.path.join(option.data_dir,'mnist_10color_jitter_var_%.03f.npy'%option.color_var))
        if self.is_train == 1:
            self.image = data_dic['train_image']
            self.label = data_dic['train_label']
        else:
            self.image = data_dic['test_image']
            self.label = data_dic['test_label']
            

        color_var = option.color_var
        self.color_std = color_var**0.5

        # image = torch.from_numpy(self.image).view(60000,-1,3)
        # mm = torch.sum(1 - (image==0).int(), dim=1)
        # colorlabel = torch.div(torch.sum(image, dim=1) / mm,32)
        # label = torch.from_numpy(self.label)
        # dd = 0
        # a = torch.sum(torch.abs(colorlabel[label == dd] - torch.mean(colorlabel[label == dd].float(), dim=0)), dim=1).sort()[0]

        # self.T = transforms.Compose([
        #                       transforms.ToTensor(),
        #                       transforms.Normalize((0.4914,0.4822,0.4465),
        #                                            (0.2023,0.1994,0.2010)),
        #                             ])

        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5,0.5,0.5),
                                             (0.5,0.5,0.5))
        self.ToPIL = transforms.Compose([
                              transforms.ToPILImage(),
                              ])


    def __getitem__(self,index):
        label = self.label[index]
        image = self.image[index]

        image = self.ToPIL(image)

        label_image = image.resize((14,14), Image.NEAREST) 

        label_image = torch.from_numpy(np.transpose(label_image,(2,0,1)).copy())
        mask_image = torch.lt(label_image.float()-0.00001, 0.) * 255
        label_image = torch.div(label_image,32)
        label_image = label_image.int() + mask_image
        label_image = label_image.long()
        colorlabel = torch.mean(label_image[label_image != 255].view(3, -1).float(), 1).long()

        image = self.toTensor(image)
        image = self.normalize(image)
        return image, label_image,  label.astype(np.long), colorlabel

        

    def __len__(self):
        return self.image.shape[0]



