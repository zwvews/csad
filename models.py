import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.init as init
import torch.nn.functional as F
import random

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

class convnet(nn.Module):
    def __init__(self):
        super(convnet, self).__init__()
        self.conv1   = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=0)
        self.relu1    = nn.ReLU(inplace=True)
        self.relu2    = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2   = nn.Conv2d(64,64, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        for m in self.children():
            weights_init_kaiming(m)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)  # 28x28
        x = self.maxpool1(x) # 14x14

        x = self.conv2(x)
        x = self.relu2(x) #14x14
        x = self.maxpool2(x) #64*4*4
        return x.view(x.shape[0], -1)

class disentangler(nn.Module):
    def __init__(self):
        super(disentangler, self).__init__()
        self.fc1 = nn.Linear(1024,128)
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(512, 128)
        for m in self.children():
            weights_init_kaiming(m)
    def forward(self, x):
        x = self.fc1(x)

        return x

class classifier(nn.Module):
    def __init__(self, numclass=10):
        super(classifier, self).__init__()
        self.fc1 = nn.Linear(128,64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(64, numclass)
        for m in self.children():
            weights_init_kaiming(m)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class biasClassifier(nn.Module):
    def __init__(self, numclass=8):
        super(biasClassifier, self).__init__()
        self.fc1r = nn.Linear(128,64)
        self.relur = nn.LeakyReLU(inplace=True)
        self.fc2r = nn.Linear(64, numclass)

        self.fc1g = nn.Linear(128,64)
        self.relug = nn.LeakyReLU(inplace=True)
        self.fc2g = nn.Linear(64, numclass)

        self.fc1b = nn.Linear(128,64)
        self.relub = nn.LeakyReLU(inplace=True)
        self.fc2b = nn.Linear(64, numclass)

        for m in self.children():
            weights_init_kaiming(m)
    def forward(self, x):

        xr = self.fc1r(x)
        xr = self.relur(xr)
        xr = self.fc2r(xr)

        xg = self.fc1g(x)
        xg = self.relug(xg)
        xg = self.fc2g(xg)

        xb = self.fc1b(x)
        xb = self.relub(xb)
        xb = self.fc2b(xb)
        return xr, xg, xb



class MI(nn.Module):
    def __init__(self, option):
        super(MI, self).__init__()
        self.option = option

        self.fc1_cls = nn.Linear(128, 128)
        self.fc2_cls = nn.Linear(128, 64)
        self.fc3_cls = nn.Linear(64, 64)

        self.fc1_bias = nn.Linear(128, 128)
        self.fc2_bias = nn.Linear(128, 64)
        self.fc3_bias = nn.Linear(64, 64)

        for m in self.children():
            weights_init_kaiming(m)
        self.knnt1 = nn.parameter.Parameter(torch.Tensor([self.option.tau]), requires_grad=True)
        self.knnt2 = nn.parameter.Parameter(torch.Tensor([self.option.tau]), requires_grad=True)
        self.negt = nn.parameter.Parameter(torch.Tensor([self.option.alpha]), requires_grad=True)
        self.post = nn.parameter.Parameter(torch.Tensor([self.option.alpha]), requires_grad=True)


    def propogation(self, A1_normalized, c, RWR=True):
        if RWR:
            Ak1 = (1 - c) * torch.inverse(torch.eye(len(A1_normalized)).cuda() - c * A1_normalized)
        else:
            Ak1 = A1_normalized
        Ak1 = (Ak1 + Ak1.t()) / 2
        Ak1 = Ak1 / torch.sum(Ak1, dim=1, keepdim=True)
        return Ak1


    def forward(self, feaClass, feaBias, label, colorlabel, adv=False):
        # get content features of target branch
        feacls = self.fc1_cls(feaClass)
        feacls = nn.functional.leaky_relu(feacls)
        feacls = self.fc2_cls(feacls)
        feacls = nn.functional.leaky_relu(feacls)
        feacls = self.fc3_cls(feacls)
        feacls = nn.functional.normalize(feacls)

        # get content features of bias branch
        feabias = self.fc1_bias(feaBias)
        feabias = nn.functional.leaky_relu(feabias)
        feabias = self.fc2_bias(feabias)
        feabias = nn.functional.leaky_relu(feabias)
        feabias = self.fc3_bias(feabias)
        feabias = nn.functional.normalize(feabias)

        # calculate similarity matrix for content features
        A = feacls.matmul(feabias.t())  # batch*batch

        # get similarity matrix for structural features
        Acls = feacls.matmul(feacls.t())
        Acls = F.softmax(self.knnt1 * Acls, dim=1)
        Acls = (Acls + Acls.t()) / 2
        Abias = feabias.matmul(feabias.t())
        Abias = F.softmax(self.knnt2 * Abias, dim=1)
        Abias = (Abias + Abias.t()) / 2

        mask = []
        for i in range(feacls.shape[0]):
            maskItem = colorlabel - colorlabel[i]
            mask.append(maskItem)
        mask = torch.stack(mask, dim=0)
        mask = torch.sum(torch.abs(mask) ** 2, dim=2)
        mask = mask < 4
        # label propogation
        c = 0.5
        Acls_prop = self.propogation(Acls, c)
        Abias_prop = self.propogation(Abias, c)
        A_clsbias = Acls_prop.matmul(torch.log(Abias_prop.t() + 1e-7)) + \
              Abias_prop.matmul(torch.log(Acls_prop.t() + 1e-7))
        A_clsbias = A_clsbias / 2

        pos = A[mask]
        pos_clsbias = A_clsbias[mask]
        neg = A[~mask]
        neg_clsbias = A_clsbias[~mask]
        wpos = self.post
        wneg = self.negt
        loss_MI = torch.log(1 + torch.sum(torch.exp(-wpos * (1*pos_clsbias+pos)))) / wpos + torch.log(
            1 + torch.sum(torch.exp(wneg * (1*neg_clsbias+neg)))) / wneg
        if adv:
            loss_advMI = - loss_MI
        else:
            loss_advMI = 0
        return loss_MI, A, loss_advMI
