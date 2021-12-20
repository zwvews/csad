# -*- coding: utf-8 -*-
import os
import json
import time
import logging


def save_option(option):
    option_path = os.path.join(option.save_dir, option.exp_name, "options.json")

    with open(option_path, 'w') as fp:
        json.dump(option.__dict__, fp, indent=4, sort_keys=True)


def logger_setting(exp_name, save_dir, debug, filename='train.log'):
    logger = logging.getLogger(exp_name)
    formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')

    log_out = os.path.join(save_dir, exp_name, filename)
    file_handler = logging.FileHandler(log_out)
    stream_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    return logger


def printandlog(str1, savefilepath):
    print(str1)
    with open(savefilepath + '.txt', 'a+') as f:
        f.write(str1)
        f.write('\n')


def _num_correct(outputs, labels, topk=1):
    _, preds = outputs.topk(k=topk, dim=1)
    preds = preds.t()
    correct = preds.eq(labels.view(1, -1).expand_as(preds))
    correct = correct.view(-1).sum()
    return correct


def _accuracy( outputs, labels):
    batch_size = labels.size(0)
    _, preds = outputs.topk(k=1, dim=1)
    preds = preds.t()
    correct = preds.eq(labels.view(1, -1).expand_as(preds))
    correct = correct.view(-1).float().sum(0, keepdim=True)
    accuracy = correct.mul_(100.0 / batch_size)
    return accuracy
