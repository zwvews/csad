# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import models as models
import numpy as np
import time
import os
import math
from tqdm import tqdm
from utils import logger_setting, printandlog, _num_correct, _accuracy


class Trainer(object):
    def __init__(self, option):
        self.option = option
        logfilename = option.exp_name
        self.logger = logger_setting(option.exp_name, option.save_dir, option.debug, logfilename)
        self.logfilename = logfilename
        self.feaExtractor = models.convnet().cuda()
        self.biasDisentangle = models.disentangler().cuda()
        self.classDisentangle = models.disentangler().cuda()
        self.biasPredictor = models.biasClassifier(option.n_class_bias).cuda()
        self.classPredictor = models.classifier(option.n_class).cuda()
        self.MI = models.MI(option).cuda()
        self.valiloader = None
        self.bias_loss = nn.CrossEntropyLoss(ignore_index=255).cuda()
        self.classification_loss = nn.CrossEntropyLoss(ignore_index=255).cuda()

        betad = (0.9, 0.999)
        self.betad = betad
        self.optim_feaextractor = optim.Adam(self.feaExtractor.parameters(), betas=self.betad, lr=self.option.lr,
                                             weight_decay=self.option.weight_decay)
        self.optim_biasDisentangle = optim.Adam(self.biasDisentangle.parameters(), betas=self.betad, lr=self.option.lr,
                                                weight_decay=self.option.weight_decay)
        self.optim_ClassDisentangle = optim.Adam(self.classDisentangle.parameters(), betas=self.betad, lr=self.option.lr,
                                                 weight_decay=self.option.weight_decay)
        self.optim_biasPredictor = optim.Adam(self.biasPredictor.parameters(), lr=self.option.lr, betas=self.betad,
                                              weight_decay=self.option.weight_decay)
        self.optim_classPredictor = optim.Adam(self.classPredictor.parameters(), lr=self.option.lr, betas=self.betad,
                                               weight_decay=self.option.weight_decay)

        self.optim_MI = optim.Adam(self.MI.parameters(), lr=self.option.lr, betas=betad,
                                   weight_decay=self.option.weight_decay)

        lr_lambda = lambda step: self.option.lr_decay_rate ** (step // self.option.lr_decay_period)

        self.scheduler_feaExt = optim.lr_scheduler.LambdaLR(self.optim_feaextractor, lr_lambda=lr_lambda, last_epoch=-1)
        self.scheduler_biasDisentangle = optim.lr_scheduler.LambdaLR(self.optim_biasDisentangle, lr_lambda=lr_lambda,
                                                                     last_epoch=-1)
        self.scheduler_classDisentangle = optim.lr_scheduler.LambdaLR(self.optim_ClassDisentangle, lr_lambda=lr_lambda,
                                                                      last_epoch=-1)
        self.scheduler_biasPredictor = optim.lr_scheduler.LambdaLR(self.optim_biasPredictor, lr_lambda=lr_lambda,
                                                                   last_epoch=-1)
        self.scheduler_classPredictor = optim.lr_scheduler.LambdaLR(self.optim_classPredictor, lr_lambda=lr_lambda,
                                                                    last_epoch=-1)
        self.scheduler_MI = optim.lr_scheduler.LambdaLR(self.optim_MI, lr_lambda=lr_lambda, last_epoch=-1)

    def _mode_setting(self, is_train=True):
        if is_train:
            self.feaExtractor.train()
            self.biasDisentangle.train()
            self.classDisentangle.train()
            self.biasPredictor.train()
            self.classPredictor.train()
            self.MI.train()
        else:
            self.feaExtractor.eval()
            self.biasDisentangle.eval()
            self.classDisentangle.eval()
            self.biasPredictor.eval()
            self.classPredictor.eval()
            self.MI.eval()

    def _optim_zero_grad(self):
        self.optim_classPredictor.zero_grad()
        self.optim_biasPredictor.zero_grad()
        self.optim_ClassDisentangle.zero_grad()
        self.optim_biasDisentangle.zero_grad()
        self.optim_feaextractor.zero_grad()

    def _optim_step(self):
        self.optim_classPredictor.step()
        self.optim_biasPredictor.step()
        self.optim_ClassDisentangle.step()
        self.optim_biasDisentangle.step()
        self.optim_feaextractor.step()

    def _train_step(self, data_loader, step):
        if step == 0:
            # pretrain MI
            for i in range(5):
                self._train_step_MI(data_loader)

        tbar = tqdm(data_loader)
        for i, (images, _, labels, colorlabel) in enumerate(tbar):
            # Step 1: train feature extractor and target prediction branch
            images = self._get_variable(images)
            labels = self._get_variable(labels)
            colorlabel = self._get_variable(colorlabel)
            self.optim_feaextractor.zero_grad()
            self.optim_ClassDisentangle.zero_grad()
            self.optim_classPredictor.zero_grad()
            fea = self.feaExtractor(images)
            fea_cls_disentangle = self.classDisentangle(fea)
            pred_label = self.classPredictor(fea_cls_disentangle)
            loss_pred = self.classification_loss(pred_label, torch.squeeze(labels))
            loss_training = 1 * loss_pred  # maybe with mutual max entropy
            loss_training.backward()
            self.optim_feaextractor.step()
            self.optim_ClassDisentangle.step()
            self.optim_classPredictor.step()

            # step 2: train bias prediction branch
            for j in range(10):
                self.optim_biasPredictor.zero_grad()
                self.optim_biasDisentangle.zero_grad()
                fea_bias_disentangle = self.biasDisentangle(fea.detach())
                pred_r, pred_g, pred_b = self.biasPredictor(fea_bias_disentangle)
                loss_pred_r = self.bias_loss(pred_r, torch.squeeze(colorlabel[:, 0]))
                loss_pred_g = self.bias_loss(pred_g, torch.squeeze(colorlabel[:, 1]))
                loss_pred_b = self.bias_loss(pred_b, torch.squeeze(colorlabel[:, 2]))
                loss_pred_bias = (loss_pred_r + loss_pred_g + loss_pred_b) / 3
                loss = loss_pred_bias
                loss.backward()
                self.optim_biasPredictor.step()
                self.optim_biasDisentangle.step()

            # step 3: train MI
            fea = self.feaExtractor(images)
            fea_cls_disentangle = self.classDisentangle(fea)
            fea_bias_disentangle = self.biasDisentangle(fea)
            for j in range(10):
                self.optim_MI.zero_grad()
                lossMI, A, _ = self.MI(fea_cls_disentangle.detach(), fea_bias_disentangle.detach(), labels, colorlabel)
                lossMI.backward()
                self.optim_MI.step()

            # step 4: adv train feature extractor
            self.optim_feaextractor.zero_grad()
            fea = self.feaExtractor(images)
            fea_cls_disentangle = self.classDisentangle(fea)
            fea_bias_disentangle = self.biasDisentangle(fea)
            loss_MI, A, advMI = self.MI(fea_cls_disentangle, fea_bias_disentangle, labels, colorlabel,
                                        adv=True)  # minimize MI or to zero
            loss = self.option.lambda_ * advMI
            loss.backward()
            self.optim_feaextractor.step()

            msg = "[TRAIN] cls loss : %.6f (epoch %d.%02d)" \
                  % (loss_training, step, int(100 * i / data_loader.__len__()))
            tbar.set_description(msg)

    def _train_step_MI(self, data_loader, MaxIteration=1000):
        for i, (images, _, labels, colorlabel) in enumerate(data_loader):
            self.optim_MI.zero_grad()
            if i > MaxIteration:
                return
            images = self._get_variable(images)
            labels = self._get_variable(labels)
            colorlabel = self._get_variable(colorlabel)
            fea = self.feaExtractor(images)
            fea_cls_disentangle = self.classDisentangle(fea).detach()
            fea_bias_disentangle = self.biasDisentangle(fea).detach()

            lossMI, A, _ = self.MI(fea_cls_disentangle, fea_bias_disentangle, labels, colorlabel)  # maximize MI

            loss = 1 * lossMI
            loss.backward()
            self.optim_MI.step()
            if (i + 10) % 50 == 0:
                msg = "[PreTRAIN] MI (epoch %d.%02d) MI Loss: %.6f" \
                      % (0, int(100 * i / data_loader.__len__()), lossMI)
                print(msg)

    def _pretrain(self, data_loader):
        # we use a larger learning rate to pretrain the model for fast convergence
        lr = 10*self.option.lr
        self.optim_feaextractor = optim.Adam(self.feaExtractor.parameters(), betas=self.betad, lr=lr,
                                             weight_decay=self.option.weight_decay)
        self.optim_biasDisentangle = optim.Adam(self.biasDisentangle.parameters(), betas=self.betad, lr=lr,
                                                weight_decay=self.option.weight_decay)
        self.optim_ClassDisentangle = optim.Adam(self.classDisentangle.parameters(), betas=self.betad, lr=lr,
                                                 weight_decay=self.option.weight_decay)
        self.optim_biasPredictor = optim.Adam(self.biasPredictor.parameters(), lr=lr, betas=self.betad,
                                              weight_decay=self.option.weight_decay)
        self.optim_classPredictor = optim.Adam(self.classPredictor.parameters(), lr=lr, betas=self.betad,
                                               weight_decay=self.option.weight_decay)
        for i in range(15):
            self._train_step_baseline(data_loader, i)
            self._validate(self.valiloader, i)
        print('baseline pretrian finished')
        for i in range(15):
            self._train_step_color(data_loader, i)
            self._validate(self.valiloader, i)
        print('bias brach pretrian finished')

        # reinitlized the optimizer
        lr = self.option.lr
        self.optim_feaextractor = optim.Adam(self.feaExtractor.parameters(), betas=self.betad, lr=lr,
                                             weight_decay=self.option.weight_decay)
        self.optim_biasDisentangle = optim.Adam(self.biasDisentangle.parameters(), betas=self.betad, lr=lr,
                                                weight_decay=self.option.weight_decay)
        self.optim_ClassDisentangle = optim.Adam(self.classDisentangle.parameters(), betas=self.betad, lr=lr,
                                                 weight_decay=self.option.weight_decay)
        self.optim_biasPredictor = optim.Adam(self.biasPredictor.parameters(), lr=lr, betas=self.betad,
                                              weight_decay=self.option.weight_decay)
        self.optim_classPredictor = optim.Adam(self.classPredictor.parameters(), lr=lr, betas=self.betad,
                                               weight_decay=self.option.weight_decay)

    def _train_step_baseline(self, data_loader, step):
        for i, (images, _, labels, colorlabel) in enumerate(data_loader):

            images = self._get_variable(images)
            labels = self._get_variable(labels)

            self._optim_zero_grad()
            pred_label = self.classPredictor(self.classDisentangle(self.feaExtractor(images)))

            loss_pred = self.classification_loss(pred_label, torch.squeeze(labels))
            loss_pred = loss_pred
            loss_pred.backward()
            self._optim_step()
            if i % self.option.log_step == 0:
                msg = "[PreTRAIN] cls loss : %.6f (epoch %d.%02d)" \
                      % (loss_pred, step, int(100 * i / data_loader.__len__()))
                print(msg)

    def _train_step_color(self, data_loader, step):
        for i, (images, _, labels, colorlabel) in enumerate(data_loader):
            images = self._get_variable(images)
            labels = self._get_variable(labels)
            colorlabel = self._get_variable(colorlabel)

            self.optim_biasDisentangle.zero_grad()
            self.optim_biasPredictor.zero_grad()
            fea = self.biasDisentangle(self.feaExtractor(images).detach())
            pred_r, pred_g, pred_b = self.biasPredictor(fea)

            loss_pred_r = self.bias_loss(pred_r, torch.squeeze(colorlabel[:,0]))
            loss_pred_g = self.bias_loss(pred_g, torch.squeeze(colorlabel[:,1]))
            loss_pred_b = self.bias_loss(pred_b, torch.squeeze(colorlabel[:,2]))
            loss_pred = (loss_pred_r+loss_pred_g+loss_pred_b)
            loss_pred.backward()
            self.optim_biasPredictor.step()
            self.optim_biasDisentangle.step()
            # self._optim_step()
            if i % self.option.log_step == 0:
                msg = "[PreTRAIN] bias cls loss : %.6f (epoch %d.%02d)" \
                      % (loss_pred, step, int(100 * i / data_loader.__len__()))
                print(msg)
                # self.logger.info(msg)

    def _validate(self, data_loader, epoch):
        self._mode_setting(is_train=False)
        total_num_correct = 0.
        total_num_correct_r = 0.
        total_num_correct_g = 0.
        total_num_correct_b = 0.
        total_num_test = 0.
        total_loss = 0.
        with torch.no_grad():
            for i, (images, color_labels, labels, colorlabel) in enumerate(data_loader):
                start_time = time.time()
                images = self._get_variable(images)
                colro_labels = self._get_variable(color_labels)
                labels = self._get_variable(labels)
                colorlabel = self._get_variable(colorlabel)

                pred_label = self.classPredictor(self.classDisentangle(self.feaExtractor(images)))
                pred_label_r, pred_label_g, pred_label_b = self.biasPredictor(
                    self.biasDisentangle(self.feaExtractor(images)))
                loss = self.classification_loss(pred_label, torch.squeeze(labels))

                batch_size = images.shape[0]
                total_num_correct += _num_correct(pred_label, labels, topk=1).item()
                total_num_correct_r += _num_correct(pred_label_r, colorlabel[:, 0], topk=1).item()
                total_num_correct_g += _num_correct(pred_label_g, colorlabel[:, 1], topk=1).item()
                total_num_correct_b += _num_correct(pred_label_b, colorlabel[:, 2], topk=1).item()
                total_loss += loss.item() * batch_size
                total_num_test += batch_size

        avg_loss = total_loss / total_num_test
        avg_acc = total_num_correct / total_num_test
        avg_acc_r = total_num_correct_r / total_num_test
        avg_acc_g = total_num_correct_g / total_num_test
        avg_acc_b = total_num_correct_b / total_num_test
        msg = "Epoch: %d EVALUATION LOSS  %.4f, ACCURACY : %.4f (%d/%d); ACC_r : %.4f ACC_g : %.4f; ACC_b : %.4f" % \
              (epoch, avg_loss, avg_acc, int(total_num_correct), total_num_test, avg_acc_r, avg_acc_g, avg_acc_b)
        printandlog(msg, self.logfilename)
        return avg_acc

    def _save_model(self, step, prefix=''):
        filename = os.path.join(self.option.save_dir, self.option.exp_name, prefix+'checkpoint_step_%04d.pth' % step)
        torch.save({
            'step': step,
            'feaExtractor': self.feaExtractor.state_dict(),
            'biasDisentangle': self.biasDisentangle.state_dict(),
            'classDisentangle': self.classDisentangle.state_dict(),
            'classPredictor': self.classPredictor.state_dict(),
            'biasPredictor': self.biasPredictor.state_dict(),
            'MI': self.MI.state_dict(),
        }, filename)
        print('checkpoint saved at: '+filename)

    def _load_model(self):
        ckpt = torch.load(self.option.checkpoint)

        self.feaExtractor.load_state_dict(ckpt['feaExtractor'])
        self.biasDisentangle.load_state_dict(ckpt['biasDisentangle'])
        self.classDisentangle.load_state_dict(ckpt['classDisentangle'])
        self.classPredictor.load_state_dict(ckpt['classPredictor'])
        self.biasPredictor.load_state_dict(ckpt['biasPredictor'])
        print(self.option.checkpoint + ' is loaded')

    def _scheduler_step(self):
        self.scheduler_biasDisentangle.step()
        self.scheduler_biasPredictor.step()
        self.scheduler_classDisentangle.step()
        self.scheduler_classPredictor.step()
        self.scheduler_feaExt.step()
        self.scheduler_MI.step()

    def train(self, train_loader, val_loader=None):
        self.valiloader = val_loader
        if self.option.checkpoint is not None:
            self._load_model()
        else:
            # pretraining
            print('no checkpoint specified, pretrain the model')
            self._pretrain(train_loader)
            self._save_model(step=0, prefix='pretrain')

        start_epoch = 0
        printandlog('start training', self.logfilename)

        # check baseline performance
        print('performance before current training')
        self._validate(val_loader, 0)

        for step in range(start_epoch, self.option.max_step):
            self._mode_setting(is_train=True)
            self._train_step(train_loader, step)

            self._scheduler_step()
            if step == 0 or step % self.option.save_step == 0 or step == (self.option.max_step - 1):
                if val_loader is not None:
                    acc = self._validate(val_loader, step)

            self._save_model(step)

    def _get_variable(self, inputs):
        if self.option.cuda:
            return Variable(inputs.cuda())
        return Variable(inputs)
