import argparse
import os
from distutils.util import strtobool

parser = argparse.ArgumentParser()

# Environment
parser.add_argument("--is_train", type=strtobool, default='true')
parser.add_argument("--tensorboard", type=strtobool, default='true')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument("--num_gpu", type=int, default=1)
parser.add_argument("--exp_dir", type=str, default="../WHAT_exp")
parser.add_argument("--exp_load", type=str, default=None)

# Data
parser.add_argument("--data_dir", type=str, default="/mnt/sda")
parser.add_argument("--data_name", type=str, default="fashion_mnist")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--rgb_range', type=int, default=1)

# Model
parser.add_argument('--uncertainty', default='normal',
                    choices=('normal', 'epistemic', 'aleatoric', 'combined'))
parser.add_argument('--in_channels', type=int, default=1)
parser.add_argument('--n_feats', type=int, default=32)
parser.add_argument('--var_weight', type=float, default=1.)
parser.add_argument('--drop_rate', type=float, default=0.2)

# Train
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--decay", type=str, default='50-100-150-200')
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--optimizer", type=str, default='rmsprop',
                    choices=('sgd', 'adam', 'rmsprop'))
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
parser.add_argument("--epsilon", type=float, default=1e-8)

# Test
parser.add_argument('--n_samples', type=int, default=25)


def save_args(obj, defaults, kwargs):
    for k,v in defaults.iteritems():
        if k in kwargs: v = kwargs[k]
        setattr(obj, k, v)


def get_config():
    config = parser.parse_args()
    config.data_dir = os.path.expanduser(config.data_dir)
    return config









import numpy as np
from torch.utils.tensorboard import SummaryWriter

from model import *
from loss import Loss
from util import make_optimizer, calc_psnr, summary


class Operator:
    def __init__(self, config, ckeck_point):
        self.config = config
        self.epochs = config.epochs
        self.ckpt = ckeck_point
        self.tensorboard = config.tensorboard
        if self.tensorboard:
            self.summary_writer = SummaryWriter(self.ckpt.log_dir, 300)

        # set model, criterion, optimizer
        self.model = Model(config)
        summary(self.model, config_file=self.ckpt.config_file)

        # set criterion, optimizer
        self.criterion = Loss(config)
        self.optimizer = make_optimizer(config, self.model)

        # load ckpt, model, optimizer
        if not config.is_train or self.ckpt.exp_load is not None :
            print("Loading model... ")
            self.load(self.ckpt)
            print(self.ckpt.last_epoch, self.ckpt.global_step)

    def train(self, data_loader):
        last_epoch = self.ckpt.last_epoch
        train_batch_num = len(data_loader['train'])

        for epoch in range(last_epoch, self.epochs):
            for batch_idx, batch_data in enumerate(data_loader['train']):
                batch_input, batch_label = batch_data
                batch_input = batch_input.to(self.config.device)
                batch_label = batch_label.to(self.config.device)

                # forward
                batch_results = self.model(batch_input)
                loss = self.criterion(batch_results, batch_input)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print('Epoch: {:03d}/{:03d}, Iter: {:03d}/{:03d}, Loss: {:5f}'
                      .format(epoch, self.config.epochs,
                              batch_idx, train_batch_num,
                              loss.item()))

                # use tensorboard
                if self.tensorboard:
                    current_global_step = self.ckpt.step()
                    self.summary_writer.add_scalar('train/loss',
                                                   loss, current_global_step)
                    self.summary_writer.add_images("train/input_img",
                                                   batch_input,
                                                   current_global_step)
                    self.summary_writer.add_images("train/mean_img",
                                                   torch.clamp(batch_results['mean'], 0., 1.),
                                                   current_global_step)

            # use tensorboard
            if self.tensorboard:
                print(self.optimizer.get_lr(), epoch)
                self.summary_writer.add_scalar('epoch_lr',
                                               self.optimizer.get_lr(), epoch)

            # test model & save model
            self.optimizer.schedule()
            self.save(self.ckpt, epoch)
            self.test(data_loader)
            self.model.train()

        self.summary_writer.close()

    def test(self, data_loader):
        with torch.no_grad():
            self.model.eval()

            total_psnr = 0.
            psnrs = []
            test_batch_num = len(data_loader['test'])
            for batch_idx, batch_data in enumerate(data_loader['test']):
                batch_input, batch_label = batch_data
                batch_input = batch_input.to(self.config.device)
                batch_label = batch_label.to(self.config.device)

                # forward
                batch_results = self.model(batch_input)
                current_psnr = calc_psnr(batch_results['mean'], batch_input)
                psnrs.append(current_psnr)
                total_psnr = sum(psnrs) / len(psnrs)
                print("Test iter: {:03d}/{:03d}, Total: {:5f}, Current: {:05f}".format(
                    batch_idx, test_batch_num,
                    total_psnr, psnrs[batch_idx]))

            # use tensorboard
            if self.tensorboard:
                self.summary_writer.add_scalar('test/psnr',
                                               total_psnr, self.ckpt.last_epoch)
                self.summary_writer.add_images("test/input_img",
                                               batch_input, self.ckpt.last_epoch)
                self.summary_writer.add_images("test/mean_img",
                                               torch.clamp(batch_results['mean'], 0., 1.),
                                               self.ckpt.last_epoch)

    def load(self, ckpt):
        ckpt.load() # load ckpt
        self.model.load(ckpt) # load model
        self.optimizer.load(ckpt) # load optimizer

    def save(self, ckpt, epoch):
        ckpt.save(epoch) # save ckpt: global_step, last_epoch
        self.model.save(ckpt, epoch) # save model: weight
        self.optimizer.save(ckpt) # save optimizer:


